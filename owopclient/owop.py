import asyncio
import collections
import logging
import random
import struct
import time
from enum import Enum
from typing import Tuple, Callable, Optional, Union, Iterable, Any

import aiohttp
import aiohttp_socks
import numpy as np
from PIL import Image, ImageFont, ImageDraw

logger = logging.getLogger("root")
logging.basicConfig(format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)s %(funcName)s] - %(message)s")
logging.captureWarnings(True)
logger.setLevel(logging.DEBUG)

TILE_SIZE = 16


class CaptchaState(Enum):
    CA_WAITING = 0
    CA_VERIFYING = 1
    CA_VERIFIED = 2
    CA_OK = 3
    CA_INVALID = 4


class Rank(Enum):
    NONE = 0
    USER = 1
    MODERATOR = 2
    ADMIN = 3

    # showPlayerList
    # showDevChat
    # revealSecrets


class Tool(Enum):
    CURSOR = 0
    MOVE = 1
    PIPETTE = 2
    ERASER = 3
    ZOOM = 4
    FILL = 5
    PASTE = 6
    EXPORT = 7
    LINE = 8
    PROTECT = 9


class API:
    '''
    Encapsulates API actions to perform against the provided websocket
    '''

    @staticmethod
    async def auth_captcha(ws, token: str):
        logger.debug(f"API auth_captcha: {token}")
        buf = "CaptchA" + token
        await ws.send_str(buf)

    @staticmethod
    async def join_world(ws, worldname: str):
        logger.debug(f"API join_world: {worldname}")
        buf = worldname.encode() + struct.pack("<H", 25565)
        await ws.send_bytes(buf)

    @staticmethod
    async def send_message(ws, message: str):
        logger.debug(f"API send_message: {message}")
        # if not self.chatBucket.can_spend(1):
        #     self._do_warn("may be rate limited")
        await ws.send_str(message + "\n")

    @staticmethod
    async def request_tile(ws, tx: int, ty: int):
        logger.debug(f"API request_tile: {tx}, {ty}")
        data = struct.pack("<ii", tx, ty)
        x = await ws.send_bytes(data)

    @staticmethod
    async def update_pixel(ws, x: int, y: int, color: Tuple[int, int, int]):
        logger.debug(f"API update_pixel: {x}, {x}, {color}")
        # distx = int(x / TILE_SIZE) - int(lastSentX / (TILE_SIZE * 16))
        # distx *= distx
        # disty = int(y / TILE_SIZE) - int(lastSentY / (TILE_SIZE * 16))
        # disty *= disty
        # dist = (distx + disty) ** 0.5
        # if dist >= 3:
        #     self._do_warn(f"distance {dist} from the cursor may be too great")
        # if not self.placeBucket.can_spend(1):
        #     self._do_warn("may be rate limited")

        r, g, b = color
        data = struct.pack("<iiBBB", x, y, r, g, b)
        await ws.send_bytes(data)

    # Mouse positions are 16x more precise than pixel positions
    @staticmethod
    async def send_updates(ws, x: int, y: int, selected_color: Tuple[int, int, int], selected_tool: Tool):
        logger.debug(f"API send_updates: {x}, {y}, {selected_color}, {selected_tool}")
        # self.lastSentX, self.lastSentY = x, y
        data = struct.pack(
            "<iiBBBB",
            x, y, *selected_color, selected_tool.value
        )
        await ws.send_bytes(data)

    # Need admin perms to set tiles, so don't know if this works yet
    @staticmethod
    async def set_tile(ws, x: int, y: int, arr: np.ndarray):
        logger.debug(f"API set_tile: {x}, {y}, {arr}")
        arr = arr.astype(np.uint8).flatten()
        data = struct.pack("<ii", x, y)
        data += arr
        await ws.send_bytes(data)


class Edit:
    '''
    Represents a single desired edit to be posted to the YWOT server
    This encompasses edit data and retry logic
    '''

    def __init__(self, pos: Tuple[int, int], color: Tuple[int, int, int]):
        self.pos = pos
        self.color = color
        self.time = 0
        self._flag = False
        self._task = None

    def __eq__(self, other):
        return self.pos == other.pos and self.color == other.color

    def __repr__(self):
        return f"Edit<{self.pos}, {self.color}>"

    def _start_retry(self, cb: Callable, period: float = 3):
        async def run():
            while True:
                cb()
                await asyncio.sleep(period)

        if self._flag:
            return
        self._flag = True
        self._task = asyncio.ensure_future(run())

    def _stop_retry(self):
        if not self._flag:
            return
        self._flag = False
        self._task.cancel()


class EditQueue:
    '''
    An edit queue manages multiple edits that each require a retry after a certain period of time.
    The order is generaly LIFO, except edit retries take precedence and are moved to the front of the queue
    '''

    def __init__(self, retry_period: float = 3):
        self.retry_period = retry_period
        self.edits = collections.deque()
        self.event = asyncio.Event()

    def __len__(self):
        t = time.time()
        length = len([e for e in self.edits if t - e.time > self.retry_period])
        return length

    def __repr__(self):
        return "EditQueue<" + ", ".join(str(e) for e in self.edits) + ">"

    async def add(self, edit: Edit):
        edit._start_retry(self.setevent,
                          self.retry_period)  # Set the edit up to be retried in the future
        self.edits.append(edit)

    def setevent(self):
        self.event.set()

    async def get(self) -> Optional[Edit]:
        '''
        Retrieves the first edit that has not timed out
        :return:
        '''
        await self.event.wait()
        t = time.time()
        waiting_edits = filter(lambda e: t - e.time > self.retry_period, self.edits)
        try:
            edit = next(waiting_edits)
        except StopIteration:
            self.event.clear()
            return
        edit.time = t
        edit._start_retry(self.setevent, self.retry_period)
        try:
            next(waiting_edits)
        except StopIteration:  # If that was the last waiting edit, mark the queue as blocking
            self.event.clear()
        return edit

    async def remove(self, *edits: Tuple[Edit]):
        '''
        Removes one or more edits from the queue
        :param edits:
        :return:
        '''
        for edit in edits:
            to_remove = [e for e in self.edits if e == edit]
            for old_edit in to_remove:
                old_edit._stop_retry()
                self.edits.remove(old_edit)
            if not self.edits:
                self.event.clear()
                return


class Worker:
    def __init__(self, world_name: str, edit_queue: EditQueue, event_queue: Optional[asyncio.Queue] = None,
                 proxy: Optional[str] = None, captcha_token: Optional[str] = None):
        self.world_name = world_name
        self.edit_queue = edit_queue
        self.event_queue = event_queue
        self.proxy = proxy
        self.session = None
        self.ws = None
        self.id = None
        self._flag = False
        self.captcha_token = captcha_token

        self._send_worker_task = None
        self._recv_worker_task = None

        self.lastSentX = 0
        self.lastSentY = 0
        self.selectedColor = (255, 255, 255)  # TODO: Does this get changed?
        self.selectedTool = Tool.CURSOR

    def __repr__(self):
        status = ("alive" if self._flag else "dead")
        return f"Worker<{status} proxy={self.proxy} color={self.selectedColor} tool={self.selectedTool.name}>"

    @property
    def alive(self) -> bool:
        return self._flag

    async def _send_worker(self):
        '''
        This worker continuously sends edits from the shared edit queue to the API via the worker's websocket
        '''
        while self._flag:
            edit = await self.edit_queue.get()
            if edit is None:
                continue

            # if random.random()<.5:
            #    print("worker simulating network dropout with", edit)
            #    continue

            (x, y), color = edit.pos, edit.color
            await API.send_updates(self.ws, x * 16 + 8, y * 16 + 8, self.selectedColor, self.selectedTool)
            await API.update_pixel(self.ws, x, y, color)
            self.lastSentX, self.lastSentY = x, y
            await asyncio.sleep(0.125)

    async def _recv_worker(self):
        '''
        This worker continuously receives incoming events from the API and posts them to the event queue if one is defined
        '''
        while self._flag:
            logger.debug("Worker trying to get message")
            msg = await self.ws.receive()
            data = msg.data
            if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                logger.info(f"Worker closing from receive loop")
                await self.close()
                return
            logger.debug(f"Worker received: {data}")
            if msg.type == aiohttp.WSMsgType.TEXT:
                if data == "You are banned. Appeal on the OWOP discord server, (https://discord.io/owop)":
                    print(f"Worker {self} is banned")
                    await self.close()
                    return
                elif data == "Sorry, but you have reached the maximum number of simultaneous connections, (3).":
                    print(f"Worker {self} reached simultaneous connection limit")
                    await self.close()
                    return

            if data[0] == 0:
                # setId
                self.id = int.from_bytes(data[1:], "little")

            elif data[0] == 5:
                logger.debug(f"Worker {self.proxy} captcha state {CaptchaState(data[1])}")
                # captcha
                if CaptchaState(data[1]) == CaptchaState.CA_OK:
                    await API.join_world(self.ws, self.world_name)
                elif CaptchaState(data[1]) in (
                        CaptchaState.CA_WAITING, CaptchaState.CA_INVALID) and self.proxy is not None:
                    pass
                    # TODO: Solve captcha!
                    # captcha_token = CaptchaSolver().solve_for_proxy(self.proxy)
                    # if captcha_token is None:
                    #     continue
                    # API.auth_captcha(self.ws, captcha_token)

            elif self.event_queue is not None:
                logger.debug(f"Worker pushing event to parent: {data}")
                await self.event_queue.put(data)

    async def connect(self, timeout: float):
        '''
        Connects the worker to the OWOP API
        '''
        if self._flag:
            return
        self._flag = True
        # self.ws = ReliableWebsocket(proxy=self.proxy)

        connector = None
        if self.proxy:
            connector = aiohttp_socks.SocksConnector.from_url(self.proxy)

        self.session = aiohttp.ClientSession(connector=connector)
        logger.debug(f"Worker connecting from '{self.proxy}'")
        try:
            self.ws = await asyncio.wait_for(self.session.ws_connect("wss://ourworldofpixels.com/" + self.world_name),
                                             timeout=timeout)
        except (aiohttp_socks.SocksError, aiohttp_socks.SocksConnectionError, aiohttp.ClientError, ConnectionError,
                asyncio.TimeoutError):
            print(f"Worker could not connect from '{self.proxy}'")
            await self.close()
            return
        logger.debug(f"Worker connected from '{self.proxy}'")
        self._recv_worker_task = asyncio.ensure_future(self._recv_worker())
        self._send_worker_task = asyncio.ensure_future(self._send_worker())

    async def close(self):
        '''
        Disconnects the worker from the OWOP API and closes its associated resources
        :return:
        '''
        self._flag = False
        self.edit_queue.setevent()
        if self._recv_worker_task is not None:
            self._recv_worker_task.cancel()
        if self._send_worker_task is not None:
            self._send_worker_task.cancel()
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()


class Client:
    '''
    A Client manages multiple workers to allow for posting events from multiple streams (the API allows for up to 4 concurrent connections per IP)
    Only one of these workers is configured to allow for the receipt of incoming API events, to eliminate incoming message duplication
    '''

    def __init__(self, worldname: str = "main", retry_period: float = 1):
        self.worldname = worldname
        self.retry_period = retry_period

        self._flag = False
        self.players = {}
        self.tiles = {}
        self.rank = None
        self.selectedColor = (255, 255, 255)
        self.selectedTool = Tool.CURSOR

        self._get_id = 0
        self._pending_gets = {}
        self._pending_get_events = {}
        self._post_id = 0
        self._pending_posts = {}
        self._pending_post_events = {}

        self._event_queue = asyncio.Queue()
        self._edit_queue = None
        self._send_workers = []
        self._recv_worker_task = None

    def __setitem__(self, key, value):
        return await self.post(key, value)

    def __getitem__(self, key):
        bounds = parse_bounds(*key)
        return await self.get(bounds)

    @property
    def alive(self) -> bool:
        '''
        Returns the connection status of the client
        '''
        return self._flag

    async def hide(self):
        '''
        Moves each worker's cursor to the map boundary
        '''
        for w in self._send_workers:
            if w.alive:
                await API.send_updates(w.ws, 2147483647, 2147483647, self.selectedColor, self.selectedTool)

    def draw_tiles(self, tx0: int, ty0: int, tx1: int, ty1: int) -> np.ndarray:
        '''
        Concatenates a set of cached tiles into one continuous array
        '''
        a = np.zeros(((ty1 - ty0) * TILE_SIZE, (tx1 - tx0) * TILE_SIZE, 3), np.uint8)
        a[:, :, 0] = 255
        a[:, :, 2] = 255  # Fill empty areas with magenta
        for aty in range(ty1 - ty0):
            for atx in range(tx1 - tx0):
                if (tx0 + atx, ty0 + aty) in self.tiles:
                    ay0, ay1 = aty * TILE_SIZE, (aty + 1) * TILE_SIZE
                    ax0, ax1 = atx * TILE_SIZE, (atx + 1) * TILE_SIZE
                    a[ay0:ay1, ax0:ax1] = self.tiles[tx0 + atx, ty0 + aty]
        return a

    async def get(self, bounds: Any) -> np.ndarray:
        '''
        Retrieves a section of the map from the API as delimited by 'bounds'
        '''
        x0, y0, x1, y1 = parse_bounds(*bounds)

        tx0, tx1 = x0 // TILE_SIZE, x1 // TILE_SIZE + 1
        ty0, ty1 = y0 // TILE_SIZE, y1 // TILE_SIZE + 1

        required_tiles = set((tx, ty) for tx in range(tx0, tx1) for ty in range(ty0, ty1))
        required_tiles -= set(self.tiles)
        if required_tiles:
            get_id, self._get_id = self._get_id, self._get_id + 1
            self._pending_get_events[get_id] = asyncio.Event()
            self._pending_gets[get_id] = required_tiles.copy()

            for tx, ty in required_tiles:
                await API.request_tile(self._send_workers[0].ws, tx, ty)

            await self._pending_get_events[get_id].wait()  # TODO: wait() -> wait(1) with self._flag check
            remaining = self._pending_gets.pop(get_id)
            for pos, color in remaining:
                await self._edit_queue.remove(Edit(pos, color))
            self._pending_get_events.pop(get_id)

        a = self.draw_tiles(tx0, ty0, tx1, ty1)

        dx0 = x0 - (tx0 * TILE_SIZE)
        dy0 = y0 - (ty0 * TILE_SIZE)
        dx1 = tx1 * TILE_SIZE - x1
        dy1 = ty1 * TILE_SIZE - y1

        return a[dy0:-dy1, dx0:-dx1]

    # TODO: make smart a class-level thing? bc if i implement per-post tile refreshes, it would have to recompute the smart edits in the post worker
    async def post(self, coord: Any, obj: Union[Image.Image, Tuple[int, int, int]], fade: bool = True):
        '''
        Sets a pixel at a certain location, or posts an image
        :param coord: The coordinate at which to post the change
        :param obj: Either a color 3-tuple or a PIL Image object
        :param fade: Whether to fade the changes in - False will proceed left to right, top to bottom
        :return:
        '''
        x0, y0, x1, y1 = parse_bounds(*coord)
        x, y = x0, y0

        # Allow the user to specify a slice as a key and color as a value (a fill operator)
        if isinstance(obj, tuple) and len(obj) == 3:
            color = obj
            a = np.empty((y1 - y0, x1 - x0, 3), np.uint8)
            a[...] = color

        elif isinstance(obj, Image.Image):
            # Convert transparent images to white background
            if obj.mode == "RGBA":
                opaque = Image.new("RGBA", obj.size, (255, 255, 255, 255))
                opaque.alpha_composite(obj)
                obj = np.asarray(opaque.convert("RGB"))
            elif obj.mode != "RGB":
                obj = obj.convert("RGB")
            a = np.asarray(obj)

        else:
            a = obj

        required_edits = []
        bg = await self.get((x, y, x + a.shape[1], y + a.shape[0]))
        for ay, ax in np.argwhere(~np.all(bg == a, axis=-1)):
            pos, color = (ax + x, ay + y), tuple(a[ay, ax])
            required_edits.append(Edit(pos, color))

        if not required_edits:
            return
        if fade:
            random.shuffle(required_edits)

        post_id, self._post_id = self._post_id, self._post_id + 1
        for edit in required_edits:
            await self._edit_queue.add(edit)
        self._edit_queue.setevent()
        self._pending_post_events[post_id] = asyncio.Event()
        self._pending_posts[post_id] = required_edits

        await self._pending_post_events[post_id].wait()
        await self._edit_queue.remove(*self._pending_posts[post_id])
        self._pending_posts.pop(post_id)
        self._pending_post_events.pop(post_id)

    def post_text(self,
                  coord: Tuple[int, int],
                  text: str,
                  font: Union[ImageFont.FreeTypeFont, str],
                  fontsize: int = 12,
                  fill: Tuple[int, int, int] = (0, 0, 0),
                  fontmode: str = "L",
                  timeout: Optional[float] = None):
        '''
        Types text at the given location with a certain font
        :param coord: The location at which to begin typing
        :param text: The text to type
        :param font: A preloaded font object or name of a font to use
        :param fontsize: If 'font' is a font name, this determines the size of the loaded font
        :param fill: The color of the typed text
        :param fontmode: A PIL font mode specifier
        :param timeout:
        :return:
        '''
        x, y = coord
        if not isinstance(font, ImageFont.FreeTypeFont):
            font = ImageFont.truetype(font, fontsize)
        text_w, text_h = 0, 0

        for line in text.replace("\t", "    ").split("\n"):
            line_w, line_h = font.getsize(line)

            # bg = Image.fromarray(self.get((x, y, x+text_w, y+text_h)))
            bg = Image.new("RGB", (line_w, line_h), color=(255, 255, 255))
            drawer = ImageDraw.Draw(bg)
            drawer.fontmode = fontmode
            drawer.text((0, 0), line, font=font, fill=fill)
            await self.post((x, y), bg, fade=False)

            text_h += line_h
            text_w = max(text_w, line_w)
            y += line_h

        return text_w, text_h

    async def _recv_worker(self):
        '''
        This worker continuously receives events from the event queue and decodes them
        :return:
        '''
        while self._flag:
            data = await self._event_queue.get()
            if not self._flag:
                break

            logger.debug(f"Client received data: {data}")

            if data[0] == 1:
                # worldUpdate

                # Cursor updates
                count = data[1]
                for i in range(count):
                    cursor_data = struct.unpack("<IiiBBBB", data[2 + i * 16: 2 + (i + 1) * 16])
                    pid, pmx, pmy, pr, pg, pb, ptool = cursor_data
                    self.players[pid] = {
                        "x": pmx,
                        "y": pmy,
                        "rgb": (pr, pg, pb),
                        "tool": Tool(ptool)
                    }

                # Tile updates
                off = 2 + count * 16
                count = struct.unpack("<H", data[off: off + 2])[0]
                for i in range(count):
                    tile_data = struct.unpack("<IiiBBB", data[off + 2 + i * 15: off + 2 + (i + 1) * 15])
                    bid, bx, by, br, bg, bb = tile_data
                    # Update pixel within this tile
                    tx, ty = bx // TILE_SIZE, by // TILE_SIZE
                    px, py = bx % TILE_SIZE, by % TILE_SIZE
                    if (tx, ty) in self.tiles:
                        # print("yuo wtf", (bx, by), (br, bg, bb))
                        await self._edit_queue.remove(Edit((bx, by), (br, bg, bb)))
                        self.tiles[tx, ty][py, px] = np.asarray((br, bg, bb))
                        # Update waiting posts
                        for post_id, post_required_edits in list(self._pending_posts.items()):
                            #    print(post_required_edits)
                            update = Edit((bx, by), (br, bg, bb))
                            if update not in post_required_edits:
                                continue
                            post_required_edits.remove(update)
                            if not post_required_edits:
                                self._pending_post_events[post_id].set()

                # Disconnects
                off += 2 + count * 15
                count = data[off]
                for dpid in struct.unpack(f"<{count}I", data[off + 1:]):
                    self.players.pop(dpid)

            elif data[0] == 2:
                # tileLoad
                tx, ty, locked = struct.unpack("<iiB", data[1:10])
                u8data = data[10:]
                u8data = decompress(u8data)
                u8data = np.asarray(u8data).reshape(16, 16, 3)
                self.tiles[tx, ty] = u8data
                for get_id, get_required_tiles in list(self._pending_gets.items()):
                    get_required_tiles.discard((tx, ty))
                    # If this pending get has had all its required tiles filled, awake listeners
                    if not get_required_tiles:
                        self._pending_get_events[get_id].set()

            elif data[0] == 3:
                # teleport
                raise Exception("unhandled packet type")
            elif data[0] == 4:
                # setRank
                self.rank = data[1]

            elif data[0] == 6:
                # setPQuota
                rate, per = struct.unpack("HH", data[1:])
                # self.placeBucket = Bucket(rate, per)

            elif data[0] == 7:
                # chunkProtected
                tx, ty, newState = struct.unpack("iiB", data[1:])
                print("tile", (ty, tx), "protected with state", newState)

            else:
                print("unhandled packet type:")
                print(repr(data))

    async def connect(self, proxies: Iterable[str] = (), timeout: Optional[float] = 10):
        '''
        Connects the client and spawns a number of workers to perform the client's desired edits
        :param proxies: Enables the user to use a proxy (or multiple) for some connections
        :param timeout:
        :return:
        '''
        self._edit_queue = EditQueue(self.retry_period)
        self._send_workers.append(Worker(self.worldname, self._edit_queue, self._event_queue))
        for i in range(0):
            self._send_workers.append(Worker(self.worldname, self._edit_queue))
        for proxy in proxies:
            for i in range(3):  # Each IP is allowed at most 4 concurrent connections
                print(f"Connecting with proxy {proxy}")
                worker = Worker(self.worldname, self._edit_queue, proxy=proxy)
                self._send_workers.append(worker)
        await asyncio.wait([w.connect(timeout) for w in self._send_workers])

        if self._flag:
            return
        self._flag = True

        self._recv_worker_task = asyncio.ensure_future(self._recv_worker())

    async def close(self):
        '''
        Disconnects the client's workers and closes the client
        :return:
        '''
        if not self._flag:
            return
        self._flag = False
        await asyncio.gather(*[worker.close() for worker in self._send_workers])
        await self._event_queue.put(None)
        await self._recv_worker_task


def decompress(u8arr: bytes) -> bytes:
    '''
    An implementation of the tile compression algorithm used by the OWOP browser client
    :param u8arr: Compressed tile data
    :return: Decompressed tile data
    '''
    originalLength = u8arr[1] << 8 | u8arr[0]
    u8decompressedarr = bytearray(originalLength)
    numOfRepeats = u8arr[3] << 8 | u8arr[2]
    offset = numOfRepeats * 2 + 4
    uptr = 0
    cptr = offset
    for i in range(numOfRepeats):
        currentRepeatLoc = (u8arr[4 + i * 2 + 1] << 8 | u8arr[4 + i * 2]) + offset
        while cptr < currentRepeatLoc:
            u8decompressedarr[uptr] = u8arr[cptr]
            uptr += 1
            cptr += 1
        repeatedNum = u8arr[cptr + 1] << 8 | u8arr[cptr]
        repeatedColorR = u8arr[cptr + 2]
        repeatedColorG = u8arr[cptr + 3]
        repeatedColorB = u8arr[cptr + 4]
        cptr += 5
        while repeatedNum:
            u8decompressedarr[uptr] = repeatedColorR
            u8decompressedarr[uptr + 1] = repeatedColorG
            u8decompressedarr[uptr + 2] = repeatedColorB
            uptr += 3
            repeatedNum -= 1
    while cptr < len(u8arr):
        u8decompressedarr[uptr] = u8arr[cptr]
        uptr += 1
        cptr += 1
    return u8decompressedarr


def parse_bounds(*args: Any) -> Tuple[int, int, int, int]:
    '''
    Parse multiple types of bounds specifications (slices, 2-tuples, 4-tuples) and normalize to a 4-tuple
    :param args:
    :return:
    '''
    if len(args) == 4:
        return args
    elif len(args) != 2:
        raise ValueError("invalid slice")

    if isinstance(args[0], slice):
        y0, y1 = args[0].start, args[0].stop
    elif isinstance(args[0], int):
        x0, x1 = args[0], args[0] + 1
    else:
        raise ValueError("invalid slice")

    if isinstance(args[1], slice):
        x0, x1 = args[1].start, args[1].stop
    elif isinstance(args[1], int):
        y0, y1 = args[1], args[1] + 1
    else:
        raise ValueError("invalid slice")

    return x0, y0, x1, y1


if __name__ != "__main__":
    logger.disabled = True

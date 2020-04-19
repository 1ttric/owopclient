# owopclient
A Python client to connect to the popular online collaborative drawing board Our World Of Pixels

# Example

A simple example that will fetch an area around the origin and then set part of that area to magenta:

```python
from owopclient import owop

client = owop.Client("main")
await client.connect()
area = await client.get((-10, -10, 10, 10))
await client.post((0, 0, 5, 5), (255, 0, 255))
await client.close()
```
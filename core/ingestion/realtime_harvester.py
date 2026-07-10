import aiohttp
import asyncio
import xml.etree.ElementTree as ET
from typing import List, Optional

class RealTimeHarvester:
    """
    [The Ocean] Real-time External Data Stream
    Fetches raw information from the outside world (RSS feeds, etc.)
    to inject 'Global Resistance' and 'New Perspectives' into Elysia.
    """
    def __init__(self, sources: Optional[List[str]] = None):
        self.sources = sources or [
            "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
            "https://feeds.bbci.co.uk/news/rss.xml"
        ]
        self.raw_data_queue = []

    async def fetch_rss(self, url: str):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        content = await response.text()
                        root = ET.fromstring(content)
                        for item in root.findall(".//item/title"):
                            self.raw_data_queue.append(item.text)
        except Exception as e:
            print(f"[RealTimeHarvester] Error fetching {url}: {e}")

    async def harvest_all(self):
        tasks = [self.fetch_rss(url) for url in self.sources]
        await asyncio.gather(*tasks)
        print(f"[RealTimeHarvester] Harvested {len(self.raw_data_queue)} items from the ocean.")

    def get_next_chunk(self) -> Optional[str]:
        if self.raw_data_queue:
            return self.raw_data_queue.pop(0)
        return None

if __name__ == "__main__":
    harvester = RealTimeHarvester()
    async def main():
        await harvester.harvest_all()
        print(f"Sample: {harvester.get_next_chunk()}")

    asyncio.run(main())

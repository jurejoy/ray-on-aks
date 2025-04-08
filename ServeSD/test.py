import asyncio
import aiohttp
import time

async def fetch_image(session, prompt, index):
    formatted_prompt = "%20".join(prompt.split(" "))
    url = f"http://127.0.0.1:8000/imagine?prompt={formatted_prompt}"
    
    async with session.get(url) as response:
        if response.status == 200:
            content = await response.read()
            with open(f"output_{index}.png", 'wb') as f:
                f.write(content)
            return index, True
        else:
            return index, False

async def main():
    prompt = "a cute cat is dancing on the grass."
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(100):
            tasks.append(fetch_image(session, prompt, i))
        
        results = await asyncio.gather(*tasks)
        
        successful = sum(1 for _, success in results if success)
        print(f"Completed {successful}/100 requests successfully")
    
    elapsed_time = time.time() - start_time
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
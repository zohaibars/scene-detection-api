from fastapi import FastAPI, Depends, HTTPException, Header, UploadFile
import os
import concurrent.futures
import asyncio
from main import main
from utils import *
import shutil
from PIL import Image
import collections
import json
import logging
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)
def convert_png_to_jpeg(png_path):
    try:
        # Open the PNG image
        png_image = Image.open(png_path)

        # Construct the path for the JPEG image in the same folder
        jpeg_file_path = os.path.splitext(png_path)[0] + '.jpg'

        # Convert and save as JPEG
        png_image.convert("RGB").save(jpeg_file_path, "JPEG")

        print("Conversion completed successfully!")
        return jpeg_file_path
    except Exception as e:
        print("An error occurred:", e)
        return None


app = FastAPI()

# Define a directory to store uploaded images
upload_dir = "uploads"
os.makedirs(upload_dir, exist_ok=True)

# Define a list of valid API keys (replace with your actual API keys)
user_api_keys = {
    "user1": "apikey1",
    "user2": "apikey2",
    # Add more users and their API keys as needed
}

def process_file(file: UploadFile):
    try:
        # Read the uploaded video file into memory
        video_content = file.file.read()

        # Create a temporary file to save the uploaded video
        video_file_path = os.path.join(upload_dir, file.filename)
        # print(video_file_path)
        with open(video_file_path, "wb") as temp_video_file:
            temp_video_file.write(video_content)

        try:
            file_type=check_file_type(video_file_path)
            # print(file_type)
            if file_type=="image":
                # print("here")
                scene = main(video_file_path)
                # print(scene)
                if scene:
                    os.remove(video_file_path)
                    transformed_data = {
                    "comulative_environment_type": scene["environment_type"],
                    "comulative_scene_categories": [category["category"] for category in scene["scene_categories"]]
                    }
                    return transformed_data
                else:
                    converted_image_path = convert_png_to_jpeg(video_file_path)
                    scene = main(converted_image_path)
                    # print("file not support")
                    os.remove(video_file_path)
                    os.remove(converted_image_path)
                    transformed_data = {
                    "comulative_environment_type": scene["environment_type"],
                    "comulative_scene_categories": [category["category"] for category in scene["scene_categories"]]
                    }
                    return transformed_data
               
            elif file_type=="video":
                frames=extract_frames(video_file_path)
                # print(frames)
                all_scene=[]
                for img in frames["all_frames"]:
                    scene=main(img)
                    all_scene.append(scene)
                shutil.rmtree(frames["folder_path"])
                # return all_scene
                # Aggregate data
                aggregated_data = {
                    "comulative_environment_type": "",
                    "comulative_scene_categories": []
                }

                # Count the occurrences of each environment type
                env_counter = collections.Counter(entry["environment_type"] for entry in all_scene)
                # Get the most common environment type
                most_common_env = env_counter.most_common(1)[0][0]
                aggregated_data["comulative_environment_type"] = most_common_env

                # Collect all scene categories
                scene_categories = [category["category"] for entry in all_scene for category in entry["scene_categories"]]
                # Get unique scene categories
                unique_scene_categories = list(set(scene_categories))
                aggregated_data["comulative_scene_categories"] = unique_scene_categories
                return aggregated_data
                
            else:
                return file_type
        except Exception as e:
            # os.remove(video_file_path)  # Remove the file even if an error occurs
            raise HTTPException(status_code=500, detail="Internal Server Error")

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Dependency to validate the API key
async def get_api_key(api_key: str = Header(None, convert_underscores=False)):
    if api_key not in user_api_keys.values():
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


@app.post("/Scene_Recognition/")
async def Scene_Recognition_endpoint(
    file: UploadFile,
    api_key: str = Depends(get_api_key),  # Require API key for this route
):
    # Create a new thread for processing each user's video
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: process_file(file)
        )
    return result


@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "ok"}

@app.get("/live")
async def live_check():
    logger.info("Live status endpoint accessed")
    return {"live": True}

@app.get("/samplelog")
async def sample_log():
    logger.info("Sample log endpoint accessed")
    return {"message": "This is a sample log message"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2008, reload=True)

# uvicorn MAPI:app --host 0.0.0.0 --port 2008 --reload
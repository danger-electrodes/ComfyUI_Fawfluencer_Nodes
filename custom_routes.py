from aiohttp import web
import os
from server import PromptServer

def get_folder_item(target_folder, folder):

    subfolder = folder.replace(target_folder, "")
    target_folder_name = os.path.basename(target_folder)

    folder_item = target_folder_name + subfolder
    folder_item = folder_item.replace("\\", "/").rstrip("/")

    if not folder_item.endswith("/"):
        folder_item = folder_item + "/"

    return folder_item

def refresh_folder(folder_name):
    folders = []
    path_to_image_folder = os.path.dirname(os.path.abspath(__file__))
    target_folder = os.path.join(path_to_image_folder, folder_name)

    main_folder = get_folder_item(target_folder, target_folder)

    folders.append(main_folder)

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        return folders
    
    for root, dirs, files in os.walk(target_folder):
        for d in dirs:
            sub_folder_abs_path = os.path.abspath(os.path.join(root, d))
            sub_folder = get_folder_item(target_folder, sub_folder_abs_path)
            folders.append(sub_folder)

    return folders

async def refresh_folders_request(request):
    """Handles API requests from JavaScript"""
    try:
        data = await request.json()
        folder_name = data['folder_name']

        folders = refresh_folder(folder_name)
        print("Received from JS:", data)

        # Example: Process the request and return a response
        response_data = {
            "status": "success",
            "message": "Image folder was refreshed.",
            "datas":{
                "folders":folders
            }
        }
        return web.json_response(response_data)

    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)


def add_custom_routes():
    @PromptServer.instance.routes.post('/refresh_folder_input')
    async def execute(request):
        return await refresh_folders_request(request)
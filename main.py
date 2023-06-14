import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def image_gallery():
    folder = 'results/figures/test'
    image_files = [file for file in os.listdir(folder) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Gallery</title>
    </head>
    <body>
        <h1>Image Gallery</h1>
        <div>
    """

    for image_file in image_files[:100]:
        image_path = os.path.join(folder, image_file)
        html += f'<img src="{image_path}" alt="Image" />\n'

    html += """
        </div>
    </body>
    </html>
    """

    return html

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

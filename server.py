from pathlib import Path
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/figures", StaticFiles(directory="results/figures"), name="figures")


@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <body>
            <a href="/train/page/1">train</a>
            <br/>
            <a href="/test/page/1">test</a>
        </body>
    </html>
    """


@app.get("/{image_folder}/page/{page_number}", response_class=HTMLResponse)
def read_page(image_folder: str, page_number: int, images_per_page: int = 100):
    start_index = (page_number - 1) * images_per_page
    end_index = start_index + images_per_page

    # Get list of all image filenames
    images = sorted(os.listdir(f'results/figures/{image_folder}'))

    # Filter the ones you want for this page
    page_images = images[start_index:end_index]

    # Generate HTML
    html_images = ''.join(f'''
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin: 10px;">
            <img src="/figures/{image_folder}/{image}">
            <p>{Path(image).stem}</p>
        </div>'''
                          for image in page_images)

    # Navigation buttons
    total_pages = len(images) // images_per_page
    if len(images) % images_per_page > 0:
        total_pages += 1

    navigation_buttons = '<div style="display: flex; justify-content: center; margin: 10px;">'
    navigation_buttons += f'<a class="button" href="/{image_folder}/page/{max(1, page_number - 1)}">Previous</a>'

    # Numbered navigation buttons
    for i in range(1, total_pages + 1):
        navigation_buttons += f' <a class="button" href="/{image_folder}/page/{i}">{i}</a> '

    navigation_buttons += f'<a class="button" href="/{image_folder}/page/{min(total_pages, page_number + 1)}">Next</a></div>'

    return f"""
    <html>
        <head>
            <style>
                .button {{
                    display: inline-block;
                    padding: 10px 15px;
                    margin: 0 5px;
                    text-decoration: none;
                    color: #000;
                    background-color: #ddd;
                    border: none;
                    border-radius: 5px;
                    transition: background-color 0.3s;
                }}
                .button:hover {{
                    background-color: #bbb;
                }}
            </style>
        </head>
        <body>
            {html_images}
            {navigation_buttons}
        </body>
    </html>
    """

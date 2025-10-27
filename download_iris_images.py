from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

queries = ["iris setosa flower", "iris versicolor flower", "iris virginica flower"]

for q in queries:
    response.download({
        "keywords": q,
        "limit": 100,
        "output_directory": "dataset",
        "image_directory": q.split()[1],  # створить setosa/versicolor/virginica
        "format": "jpg",
        "size": "medium",
        "type": "photo",
    })

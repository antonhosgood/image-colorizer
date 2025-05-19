import time
from os import PathLike
from typing import AnyStr, List, Optional

import requests


class LoremPicsum:
    """A utility class to interact with the Lorem Picsum API for downloading placeholder images."""

    API = "https://picsum.photos"
    PAGE_LIMIT = 100

    def __init__(self, width: int, height: int) -> None:
        """
        Initializes an instance of the LoremPicsum class with specific image dimensions.

        Args:
            width: The width of images to be downloaded.
            height: The height of images to be downloaded.
        """
        self.width = width
        self.height = height

    def download_image(
        self,
        path: PathLike[AnyStr] | AnyStr,
        image_id: Optional[int] = None,
        grayscale: bool = False,
        blur: Optional[int] = None,
        retries: int = 3,
    ) -> None:
        """
        Downloads a JPEG image from the Lorem Picsum API and saves it to a file.

        Args:
            path: The path the output image will be saved to.
            image_id: The ID of the image to download (default is None). If not provided, a random image will be
              downloaded.
            grayscale: Whether to download the image in grayscale (default is False).
            blur: The blur level of the image (default is None). If provided, it should be an integer between 1 and 10
              indicating the level of blur.
            retries: The number of times to retry the request (default is 3).

        Raises:
            requests.exceptions.RequestException: If the HTTP request to download the image fails.
        """
        url = self.build_url(image_id)

        params = {
            "grayscale": True if grayscale else None,
            "blur": blur if blur else None,
        }

        for attempt in range(retries + 1):
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException:
                if attempt == retries:
                    raise
                time.sleep(1)

        with open(path, "wb") as f:
            f.write(response.content)

    def build_url(self, image_id: Optional[int] = None) -> str:
        """
        Builds the URL for downloading an image from the Lorem Picsum API.

        Args:
            image_id: The ID of the image to be included in the URL (default is None). If provided, it will specify a
              particular image.

        Returns:
            str: The complete URL for the image download.
        """
        url = self.API

        if image_id is not None:
            url += f"/id/{image_id}"

        url += f"/{self.width}/{self.height}"
        return url

    @staticmethod
    def get_image_ids() -> List[int]:
        """
        Returns a list of ids of images available in the Lorem Picsum API.

        When downloading all images, iterate through the returned image IDs instead of a range as some IDs are missing.

        Raises:
            requests.exceptions.RequestException: If the HTTP request to retrieve the image list fails.
        """
        ids = []
        page = 1
        url = f"{LoremPicsum.API}/v2/list"

        while True:
            response = requests.get(
                url, params={"page": page, "limit": LoremPicsum.PAGE_LIMIT}
            )
            response.raise_for_status()

            results = response.json()
            if len(results) == 0:
                return ids

            ids += [int(x["id"]) for x in results]
            page += 1

from io import BytesIO
from typing import List

import numpy as np
import telegram
import wandb
from PIL import Image
from pytorch_lightning.utilities import rank_zero_only


class Bot:
    """Bot for sending logs to telegram_bot chat."""

    def __init__(self, token: str, chat_ids: List[int], is_debug: bool = False):
        """Init bot"""
        if is_debug or token is None:
            self.bot = None
        else:
            self.bot = telegram.Bot(token=token)
        self.chat_ids = chat_ids

    def __image_to_bytes(self, image: wandb.Image) -> BytesIO:
        image = np.array(image.to_data_array()).astype(np.uint8)
        image_io = BytesIO()
        Image.fromarray(image).save(image_io, 'PNG')
        image_io.seek(0)

        return image_io

    def __send_images(self, images: List[wandb.Image], text: str, chat_id: int):
        media = [
            telegram.InputMediaPhoto(
                self.__image_to_bytes(img), caption=text if index % 10 == 0 else None
            )
            for index, img in enumerate(images)
        ]

        for i in range(0, len(media), 10):
            self.bot.send_media_group(
                chat_id=chat_id,
                media=media[i:i + 10]
            )

    @rank_zero_only
    def send_images(self, images: List[wandb.Image], text: str):
        """Send images to bot."""
        if self.bot is None:
            return

        for chat_id in self.chat_ids:
            self.__send_images(images, text, chat_id)

    @rank_zero_only
    def send_message(self, text: str):
        """Send message to bot."""
        if self.bot is None:
            return

        if len(text) > 4096:
            text = text[-4096:]

        for chat_id in self.chat_ids:
            self.bot.send_message(chat_id=chat_id, text=text, parse_mode='html')

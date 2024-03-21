import asyncio
import logging
import sys
from os import getenv

from aiogram import Bot, Dispatcher, F, Router, types
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    FSInputFile,
)
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.utils.markdown import hbold

from generator import Generator

TOKEN = getenv("BOT_TOKEN")

dp = Dispatcher()
loop = asyncio.get_event_loop()
generator = Generator(num_inference_steps=20)
tasks = []


class States(StatesGroup):
    text2img = State()
    img2img = State()
    imgtext2img_wait_text = State()
    imgtext2img_wait_pic = State()
    audiotext2img_wait_text = State()
    audiotext2img_wait_audio = State()
    audioimg2img_wait_img = State()
    audioimg2img_wait_audio = State()
    imgimg2img_wait_pic1 = State()
    imgimg2img_wait_pic2 = State()


async def send_generated(message: Message, filename):
    result_img = FSInputFile(filename)
    await message.answer_photo(
        photo=result_img,
        caption="generated picture",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="text to image"),
                    KeyboardButton(text="image to image"),
                ],
                [
                    KeyboardButton(text="img+text to image"),
                    KeyboardButton(text="img+img to image"),
                ],
                [
                    KeyboardButton(text="audio+text to image"),
                    KeyboardButton(text="audio+image to image"),
                ],
            ],
            resize_keyboard=True,
        ),
    )


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """

    logging.error("Start cmd")
    await message.answer(
        f"Choose your fighter",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="text to image"),
                    KeyboardButton(text="image to image"),
                ],
                [
                    KeyboardButton(text="img+text to image"),
                    KeyboardButton(text="img+img to image"),
                ],
                [
                    KeyboardButton(text="audio+text to image"),
                    KeyboardButton(text="audio+image to image"),
                ],
            ],
            resize_keyboard=True,
        ),
    )


@dp.message(Command("home"))
@dp.message(F.text.casefold() == "home")
async def cancel_handler(message: Message, state: FSMContext) -> None:
    """
    Allow user to cancel any action
    """
    current_state = await state.get_state()
    if current_state is None:
        return
    await state.clear()
    await message.answer(
        f"Choose your fighter",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="text to image"),
                    KeyboardButton(text="image to image"),
                ],
                [
                    KeyboardButton(text="img+text to image"),
                    KeyboardButton(text="img+img to image"),
                ],
                [
                    KeyboardButton(text="audio+text to image"),
                    KeyboardButton(text="audio+image to image"),
                ],
            ],
            resize_keyboard=True,
        ),
    )


############################### image+image -> image #####################################
@dp.message(F.text == "img+img to image")
async def imgtext2img_start_handler(message: Message, state: FSMContext) -> None:
    """
    image + img -> image generation waiting text
    """
    logging.info("Image+Text to image state, waiting text")
    await state.set_state(States.imgimg2img_wait_pic1)
    await message.answer(f"Send your first image", reply_markup=ReplyKeyboardRemove())


@dp.message(States.imgimg2img_wait_pic1)
async def imgtext2img_got_img(message: Message, state: FSMContext) -> None:
    """
    image + image -> image generation got both imgs
    """

    current_state = await state.get_state()
    if message.photo is None or len(message.photo) == 0:
        await state.clear()
        await message.answer(f"Provide an image")
        return
    user_id = ""
    if message.from_user.id is not None:
        user_id = str(message.from_user.id)
    input_filename1 = f"input/input_picture_{user_id}_1.png"
    await message.bot.download(
        file=message.photo[-1].file_id, destination=input_filename1
    )
    logging.info(f"Image+Image to image state got images")
    await state.set_state(States.imgimg2img_wait_pic2)
    await message.answer(
        f"ok, got your picture 1. Now give me picture 2."
    )


@dp.message(States.imgimg2img_wait_pic2)
async def imgtext2img_gen_handler(message: Message, state: FSMContext) -> None:
    """
    image + image -> image generation got both imgs
    """

    current_state = await state.get_state()
    if message.photo is None or len(message.photo) == 0:
        await state.clear()
        await message.answer(f"Provide an image")
        return
    user_id = ""
    if message.from_user.id is not None:
        user_id = str(message.from_user.id)
    input_filename2 = f"input/input_picture_{user_id}_2.png"
    input_filename1 = f"input/input_picture_{user_id}_1.png"
    await message.bot.download(
        file=message.photo[-1].file_id, destination=input_filename2
    )
    logging.info(f"Image+Image to image state got images")
    await state.clear()
    await message.answer(
        f"ok, got your pictures. Hold my beer (for about 15 minutes)."
    )
    logging.info("generation started")
    generator.imageimage2image(input_filename2, input_filename1, f"generated_{user_id}")
    
    logging.info("generation ended")
    await send_generated(message, f"output/generated_{user_id}.png")


############################### image+audio -> image #####################################
@dp.message(F.text == "audio+image to image")
async def audioimg2img_start_handler(message: Message, state: FSMContext) -> None:
    """
    audio + image -> image generation waiting image
    """
    logging.info("Audio+Image to image state, waiting image")
    await state.set_state(States.audioimg2img_wait_img)
    await message.answer(f"Send your image first", reply_markup=ReplyKeyboardRemove())


@dp.message(States.audioimg2img_wait_img)
async def audioimg2img_text_handler(message: Message, state: FSMContext) -> None:
    """
    audio + image -> image generation got image
    """
    current_state = await state.get_state()
    if message.photo is None or len(message.photo) == 0:
        await state.clear()
        await message.answer(f"Provide an image")
        return
    user_id = ""
    if message.from_user.id is not None:
        user_id = str(message.from_user.id)
    input_filename1 = f"input/input_picture_{user_id}.png"
    await message.bot.download(
        file=message.photo[-1].file_id, destination=input_filename1
    )
    logging.info(f"Audio+Image to image state got image")
    await state.set_state(States.audioimg2img_wait_audio)
    await message.answer(
        f"ok, got your picture. Now give me .wav file."
    )

@dp.message(States.audioimg2img_wait_audio)
async def audioimg2img_gen_handler(message: Message, state: FSMContext) -> None:
    """
    audio + image -> image generation got audio, image
    """

    current_state = await state.get_state()
    file_id = message.audio.file_id
    if file_id is None:
        await state.clear()
        await message.answer(f"Provide a file")
        return
    user_id = ""
    if message.from_user.id is not None:
        user_id = str(message.from_user.id)
    input_filename = f"input/input_audio_{user_id}.wav"
    img_filename = f"input/input_picture_{user_id}.png"
    await message.bot.download(
        file=file_id, destination=input_filename
    )
    logging.info(f"Audio+Image to image state got image and image")
    await state.clear()
    await message.answer(
        f"ok, got your audio and image. Hold my beer (for about 15 minutes)."
    )
    logging.info("generation started")
    generator.audioimage2image(img_filename, input_filename, f"generated_{user_id}")
    logging.info("generation ended")
    await send_generated(message, f"output/generated_{user_id}.png")

    
############################### text+audio -> image #####################################
@dp.message(F.text == "audio+text to image")
async def audiotext2img_start_handler(message: Message, state: FSMContext) -> None:
    """
    audio + text -> image generation waiting text
    """
    logging.info("Audio+Text to image state, waiting text")
    await state.set_state(States.audiotext2img_wait_text)
    await message.answer(f"Send your text first", reply_markup=ReplyKeyboardRemove())


@dp.message(States.audiotext2img_wait_text)
async def audiotext2img_text_handler(message: Message, state: FSMContext) -> None:
    """
    audio + text -> image generation got text
    """
    current_state = await state.get_state()
    if message.text is None:
        await state.clear()
        await message.answer(f"Provide non-empty text")
        return

    await state.set_state(States.audiotext2img_wait_audio)
    await state.update_data(text=message.text)
    await message.answer(
        f"Ok, got '{message.text}'. Now send a .wav file",
        reply_markup=ReplyKeyboardRemove(),
    )


@dp.message(States.audiotext2img_wait_audio)
async def audiotext2img_gen_handler(message: Message, state: FSMContext) -> None:
    """
    audio + text -> image generation got audio, text
    """

    current_state = await state.get_state()
    file_id = message.audio.file_id
    if file_id is None:
        await state.clear()
        await message.answer(f"Provide a file")
        return
    user_id = ""
    if message.from_user.id is not None:
        user_id = str(message.from_user.id)
    input_filename = f"input/input_audio_{user_id}.wav"
    await message.bot.download(
        file=file_id, destination=input_filename
    )
    data = await state.get_data()
    text = data["text"]
    logging.info(f"Audio+Text to image state got image and text '{text}'")
    await state.clear()
    await message.answer(
        f"ok, got your audio and text '{text}'. Hold my beer (for about 15 minutes)."
    )
    logging.info("generation started")
    generator.audiotext2image(text, input_filename, f"generated_{user_id}")
    logging.info("generation ended")
    await send_generated(message, f"output/generated_{user_id}.png")

    
############################### text+image -> image #####################################
@dp.message(F.text == "img+text to image")
async def imgtext2img_start_handler(message: Message, state: FSMContext) -> None:
    """
    image + text -> image generation waiting text
    """
    logging.info("Image+Text to image state, waiting text")
    await state.set_state(States.imgtext2img_wait_text)
    await message.answer(f"Send your text first", reply_markup=ReplyKeyboardRemove())


@dp.message(States.imgtext2img_wait_text)
async def imgtext2img_text_handler(message: Message, state: FSMContext) -> None:
    """
    image + text -> image generation got text
    """
    current_state = await state.get_state()
    if message.text is None:
        await state.clear()
        await message.answer(f"Provide non-empty text")
        return

    await state.set_state(States.imgtext2img_wait_pic)
    await state.update_data(text=message.text)
    await message.answer(
        f"Ok, got '{message.text}'. Now send a picture",
        reply_markup=ReplyKeyboardRemove(),
    )


@dp.message(States.imgtext2img_wait_pic)
async def imgtext2img_gen_handler(message: Message, state: FSMContext) -> None:
    """
    image + text -> image generation got img, text
    """

    current_state = await state.get_state()
    # file_id = message.document.file_id
    if message.photo is None or len(message.photo) == 0:
        await state.clear()
        await message.answer(f"Provide an image")
        return
    user_id = ""
    if message.from_user.id is not None:
        user_id = str(message.from_user.id)
    input_filename = f"input/input_picture_{user_id}.png"
    await message.bot.download(
        file=message.photo[-1].file_id, destination=input_filename
    )
    data = await state.get_data()
    text = data["text"]
    logging.info(f"Image+Text to image state got image and text '{text}'")
    await state.clear()
    await message.answer(
        f"ok, got your picture and text '{text}'. Hold my beer (for about 15 minutes)."
    )
    logging.info("generation started")
    generator.imagetext2image(text, input_filename, f"generated_{user_id}")
    logging.info("generation ended")
    await send_generated(message, f"output/generated_{user_id}.png")

    
############################### image -> image #####################################
@dp.message(F.text == "image to image")
async def img2img_start_handler(message: Message, state: FSMContext) -> None:
    """
    image -> image generation waiting text
    """
    logging.info("Image2image state")
    await state.set_state(States.img2img)
    await message.answer(f"Send your image", reply_markup=ReplyKeyboardRemove())


@dp.message(States.img2img)
async def img2img_gen_handler(message: Message, state: FSMContext) -> None:
    """
    image -> image generation
    """
    current_state = await state.get_state()
    if message.photo is None or len(message.photo) == 0:
        await state.clear()
        await message.answer(f"Provide an image")
        return
    user_id = ""
    if message.from_user.id is not None:
        user_id = str(message.from_user.id)
    input_filename = f"input/input_picture_{user_id}.png"
    await message.bot.download(
        file=message.photo[-1].file_id, destination=input_filename
    )

    logging.info("Image2image state got prompt")
    await state.clear()
    await message.answer(f"ok, got your picture. Hold my beer (for about 15 minutes).")
    logging.info("generation started")
    generator.image2image(input_filename, f"generated_{user_id}")
    logging.info("generation ended")
    await send_generated(message, f"output/generated_{user_id}.png")


############################### text -> image #####################################
@dp.message(F.text == "text to image")
async def text2img_start_handler(message: Message, state: FSMContext) -> None:
    """
    text -> image generation waiting text
    """
    logging.info("Text2image state")
    await state.set_state(States.text2img)
    await message.answer(f"Send your text", reply_markup=ReplyKeyboardRemove())


@dp.message(States.text2img)
async def text2img_gen_handler(message: Message, state: FSMContext) -> None:
    """
    text -> image generation
    """
    current_state = await state.get_state()
    if message.text is None:
        await state.clear()
        await message.answer(f"Provide non-empty text")
        return
    user_id = ""
    if message.from_user.id is not None:
        user_id = str(message.from_user.id)

    logging.info("Text2image state got prompt")
    await state.clear()
    await message.answer(f"ok, got '{message.text}'. Hold my beer (for about 15 minutes).")
    logging.info("generation started")
    generator.text2image(message.text, f"generated_{user_id}")
    logging.info("generation ended")
    await send_generated(message, f"output/generated_{user_id}.png")


async def bot_shutdown():
    logging.error("stopping")
    dp.stop_polling()
    await dp.wait_closed()
    await bot.close()
    logging.error("stopped")


async def main() -> None:
    logging.error("almost started")
    bot = Bot(TOKEN, parse_mode=ParseMode.HTML)
    logging.error("almost almost started")
    task = loop.create_task(dp.start_polling(bot))
    logging.error("almost almost almost started")
    tasks.append(task)
    for ttl in range(240):
        await asyncio.sleep(60)
        logging.error(getenv("STOP"))
        if getenv("STOP") == "1":
            break

    logging.error("sleep finished")
    task.cancel()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        filename="logs.txt",
        filemode="a",
    )
    logging.error("starting")
    try:
        loop.run_until_complete(main())
    except Exception as e:
        logging.error(str(e) + ". Caught error. Canceling tasks...")
        dp.stop_polling()
        bot.close()
        for task in tasks:
            task.cancel()
        for task in tasks:
            task.exception()
    finally:
        logging.error("Finishing...")
        loop.close()

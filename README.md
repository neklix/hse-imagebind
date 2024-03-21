Для запуска бота выполните следующий набор действий:

```sh
cd src
git clone https://github.com/facebookresearch/ImageBind.git
pip install -r /home/jupyter/datasphere/project/hse-cv/project/ImageBind/requirements.txt
pip install torchaudio
pip install diffusers
cp -r ImageBind/bpe .
export BOT_TOKEN=???
python3 main.py
```
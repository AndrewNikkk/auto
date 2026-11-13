import logging
import os
from pathlib import Path
import queue
import threading
from app.config import Settings
from app.pipeline import Pipeline
from app.schema import Params


def music_choices(directory: Path):
    return [('Без музыки', '')] + [(p.name, str(p.resolve())) for p in sorted(directory.glob('*'))
                                   if p.is_file() and p.suffix.lower() in {'.mp3','.m4a','.wav','.ogg'}]


def build_ui():
    import gradio as gr
    settings = Settings.load()
    pipeline = Pipeline(settings)
    def generate(prompt,kind,languages,duration,style,seed,sources,music):
        messages = queue.Queue()
        results = {}
        def task():
            try:
                params = Params(prompt=prompt,content_type=kind,languages=languages.split(','),duration=int(duration),
                                style=style,seed=int(seed),source_urls=sources.split(),music=Path(music) if music else None)
                results['outputs'] = pipeline.run(params,progress=messages.put)
            except Exception as error:
                results['error'] = str(error)
            finally:
                messages.put(None)
        thread = threading.Thread(target=task,daemon=False)
        thread.start()
        while True:
            message = messages.get()
            if message is None:
                break
            yield message,None,None
        thread.join()
        if 'error' in results:
            raise gr.Error(results['error'])
        outputs = results['outputs']
        yield 'Готово',str(outputs['ru']) if 'ru' in outputs else None,str(outputs['en']) if 'en' in outputs else None
    with gr.Blocks(title='AI Shorts Generator') as demo:
        gr.Markdown('# AI Shorts Generator\nЛокальная генерация. Для news вставьте полный текст новости; ссылки автоматически не проверяются.')
        prompt = gr.Textbox(label='Промпт / текст новости',lines=6)
        with gr.Row():
            kind = gr.Dropdown(['auto','what_if','science','facts','story','news'],value='auto',label='Тип')
            languages = gr.Dropdown([('RU','ru'),('EN','en'),('RU+EN','ru,en')],value='ru',label='Языки')
            duration = gr.Dropdown([30,45,60],value=45,label='Целевая длительность, с')
        with gr.Row():
            style = gr.Dropdown(['cinematic','realistic','illustrated'],value='cinematic',label='Стиль')
            seed = gr.Number(value=42,precision=0,label='FLUX seed')
        sources = gr.Textbox(label='URL источников (через пробел, без скачивания)')
        music = gr.Dropdown(music_choices(settings.music_dir),value='',label='Музыка (только лицензированные файлы)')
        refresh = gr.Button('Обновить музыку')
        refresh.click(lambda: gr.Dropdown(choices=music_choices(settings.music_dir),value=''),outputs=music)
        demo.load(lambda: gr.Dropdown(choices=music_choices(settings.music_dir),value=''),outputs=music)
        button = gr.Button('Generate',variant='primary')
        status = gr.Textbox(label='Progress')
        with gr.Row():
            ru = gr.Video(label='RU',format='mp4')
            en = gr.Video(label='EN',format='mp4')
        button.click(generate,inputs=[prompt,kind,languages,duration,style,seed,sources,music],outputs=[status,ru,en],concurrency_limit=1)
    return demo


def main():
    import signal
    logging.basicConfig(level=logging.INFO)
    from app.gpu import GPUProcessManager
    def shutdown(signum, frame):
        GPUProcessManager.close_all()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    build_ui().queue(default_concurrency_limit=1).launch(server_name=os.getenv('GRADIO_BIND','127.0.0.1'),
            server_port=int(os.getenv('GRADIO_PORT','7860')),share=False,show_error=True,
            allowed_paths=[str(Settings.load().output_dir.resolve())])

if __name__ == '__main__':
    main()

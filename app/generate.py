import argparse
import logging
from pathlib import Path
from app.pipeline import Pipeline
from app.schema import Params


def main():
    parser = argparse.ArgumentParser(description='Local AI Shorts Generator')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--languages', choices=['ru','en','ru,en'], default='ru')
    parser.add_argument('--duration', type=int, choices=[30,45,60], default=45)
    parser.add_argument('--type', dest='content_type', choices=['auto','what_if','science','facts','story','news'], default='auto')
    parser.add_argument('--style', choices=['cinematic','realistic','illustrated'], default='cinematic')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--music', type=Path)
    parser.add_argument('--source-url', action='append', default=[])
    parser.add_argument('--run-id', help='Reuse a previous run ID for file-based resume')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    values = vars(args).copy()
    run_id = values.pop('run_id')
    values['languages'] = args.languages.split(',')
    values['source_urls'] = values.pop('source_url')
    try:
        outputs = Pipeline().run(Params(**values),run_id,progress=lambda m: print(m,flush=True))
    except (Exception, KeyboardInterrupt) as error:
        parser.exit(1,f'Generation failed: {error}\n')
    for language,path in outputs.items():
        print(f'{language}: {path}')

if __name__ == '__main__':
    main()

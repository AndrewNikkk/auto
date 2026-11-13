import pytest
from app.schema import Script

@pytest.fixture
def script():
    return Script(title='Test',language='ru',scenes=[dict(narration=f'Текст сцены {i}.',description='scene',
                  visual_prompt=f'A cinematic ship scene {i}',duration_seconds=4,key_fact=f'Fact {i}') for i in range(8)])

@pytest.fixture
def settings(tmp_path):
    from app.config import Settings
    return Settings(output_dir=tmp_path,comfy_dir=tmp_path/'comfy',hf_home=tmp_path/'hf')

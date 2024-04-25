import datasets
import json

_NAME = 'RuNNE'
_CITATION = '''
@article{Artemova2022runne,
    title={{RuNNE-2022 Shared Task: Recognizing Nested Named Entities}},
    author={Artemova, Ekaterina and Zmeev, Maksim and Loukachevitch,
            Natalia and Rozhkov, Igor and Batura, Tatiana and Braslavski,
            Pavel and Ivanov, Vladimir and Tutubalina, Elena},
    journal={Computational Linguistics and Intellectual Technologies:
             Proceedings of the International Conference "Dialog"},
    year={2022}
}
'''.strip()
_DESCRIPTION = 'A Russian Dataset with Nested Named Entities'
_HOMEPAGE = 'https://github.com/dialogue-evaluation/RuNNE'
_VERSION = '1.0.0'


class RuNNEBuilder(datasets.GeneratorBasedBuilder):
    _DATA_URLS = {
        'train': 'data/train.jsonl',
        'test': 'data/test.jsonl',
        'dev': 'data/dev.jsonl'
    }
    _ENTITY_TYPES_URLS = {
        'ent_types': 'ent_types.txt'
    }
    VERSION = datasets.Version(_VERSION)
    BUILDER_CONFIGS = [
        datasets.BuilderConfig('data',
                               version=VERSION,
                               description='Data'),
        datasets.BuilderConfig('ent_types',
                               version=VERSION,
                               description='Entity types list')
    ]
    DEFAULT_CONFIG_NAME = 'data'

    def _info(self) -> datasets.DatasetInfo:
        if self.config.name == 'data':
            features = datasets.Features({
                'id': datasets.Value('int32'),
                'text': datasets.Value('string'),
                'entities': datasets.Sequence(datasets.Value('string'))
            })
        else:
            features = datasets.Features({
                'type': datasets.Value('string')
            })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        if self.config.name == 'data':
            files = dl_manager.download(self._DATA_URLS)
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={'filepath': files['train']},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={'filepath': files['test']},
                ),
                datasets.SplitGenerator(
                    name='dev',
                    gen_kwargs={'filepath': files['dev']},
                ),
            ]
        else:
            files = dl_manager.download(self._ENTITY_TYPES_URLS)
            return [datasets.SplitGenerator(
                name='ent_types',
                gen_kwargs={'filepath': files['ent_types']},
            )]

    def _generate_examples(self, filepath):
        if self.config.name == 'data':
            with open(filepath, encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    yield doc['id'], doc
        else:
            with open(filepath, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    entity_type = line.strip()
                    if entity_type:
                        yield i, {'type': entity_type}

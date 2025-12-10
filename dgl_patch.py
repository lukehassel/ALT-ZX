import sys
import types


def create_mock_module(name, attrs=None):
    module = types.ModuleType(name)
    if attrs:
        for key, value in attrs.items():
            setattr(module, key, value)
    sys.modules[name] = module
    return module

datapipes_iter = create_mock_module('torchdata.datapipes.iter')
datapipes_iter.IterDataPipe = type('IterDataPipe', (), {})
datapipes_iter.Mapper = type('Mapper', (), {})
datapipes_iter.Filter = type('Filter', (), {})
datapipes_iter.Collator = type('Collator', (), {})
datapipes_iter.IterableWrapper = type('IterableWrapper', (), {})

datapipes = create_mock_module('torchdata.datapipes')
datapipes.iter = datapipes_iter

dataloader2 = create_mock_module('torchdata.dataloader2')
dataloader2_graph = create_mock_module('torchdata.dataloader2.graph')
dataloader2.graph = dataloader2_graph

import dgl.graphbolt
original_load = dgl.graphbolt.load_graphbolt
def patched_load_graphbolt():
    try:
        original_load()
    except (FileNotFoundError, ImportError, OSError):
        pass
dgl.graphbolt.load_graphbolt = patched_load_graphbolt

import dgl
print(f"DGL {dgl.__version__} imported successfully with compatibility patches")


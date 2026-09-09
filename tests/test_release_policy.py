"""The public Graph release trains user models, without pretrained downloads."""
import pytest
from spatial_axis_gat.cli import main


@pytest.mark.parametrize('command', ['download', 'models'])
def test_pretrained_commands_are_not_offered(command):
    with pytest.raises(SystemExit) as result:
        main([command])
    assert result.value.code == 2

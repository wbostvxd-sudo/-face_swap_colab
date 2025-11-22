from typing import Optional

METADATA =\
{
	'name': 'FaceFusion',
	'description': 'Industry leading face manipulation platform',
	'version': '3.5.1',
	'license': 'OpenRAIL-AS',
	'author': 'Henry Ruhs',
	'url': 'https://image_processor.io'
}


def get(key : str) -> Optional[str]:
	return METADATA.get(key)

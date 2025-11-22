import os
import sys

from image_processor.types import AppContext


def detect_app_context() -> AppContext:
	frame = sys._getframe(1)

	while frame:
		if os.path.join('image_processor', 'jobs') in frame.f_code.co_filename:
			return 'cli'
		if os.path.join('image_processor', 'uis') in frame.f_code.co_filename:
			return 'ui'
		frame = frame.f_back
	return 'cli'

INPUT_SCHEMA = {
    'image_url': {
        'type': str,
        'required': True,
    },
    'prompt': {
        'type': str,
        'required': True,
    },
    'max_new_tokens': {
        'type': int,
        'required': False,
        'default': 50
    },
}

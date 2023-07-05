from django import templates

register = templates.Library()

@register.filter
def get_value(dictionary, key):
    return dictionary.get(key, '')

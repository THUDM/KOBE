import re


def get_chinese(context):
    filtrate = re.compile(u'[^\u4E00-\u9FA5a-zA-Z0-9！？，。；、,.!?;]')
    context = filtrate.sub(r'', context)
    return context

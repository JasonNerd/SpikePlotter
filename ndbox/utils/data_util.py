import re


def dict2str(data, rank=0):
    """
    Print the data dictionary structure

    :param data: dict. The dictionary data.
    :param rank: int. The rank of data in original dictionary.
    :return: The print string.
    """

    msg = ''
    if not isinstance(data, dict):
        return ''
    for key, value in data.items():
        if not isinstance(value, dict) or len(value) == 0:
            msg += '|   ' * rank + str(key) + '\n'
        else:
            msg += '|   ' * rank + str(key) + '/\n'
            msg += dict2str(value, rank + 1)
    return msg


def split_string(string, symbols):
    out = re.split(symbols, string)
    for inx, item in enumerate(out):
        out[inx] = item.strip()
    return out

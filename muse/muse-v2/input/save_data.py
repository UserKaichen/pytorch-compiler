import struct as st

def save_data(data, path, is_act=False, to_int=False, to_hex=False, output_dir=None, q=0.0):
    def convert_hex(x):
        return '%X' % st.unpack('H', st.pack('e', x))

    if to_int:
        data = data.int()
    elif to_hex:
        data = data.half()
    elif is_act:
        data = data.mul(2**q).round().int()
    data = data.view(-1).detach().numpy()

    print(f'Saving {path}')
    path = f'{output_dir}/{path}'

    if to_hex:
        s = '\n'.join(f'{convert_hex(num)}' for num in data)
    else:
        s = '\n'.join(str(num) for num in data)

    with open(f'{path}.txt', 'w') as f:
        print(s, file=f)


def append_lib(file, content):
    with open(file, 'a') as dest:
        dest.write(content)
import zipfile

def unzip(source_file, dest_path):
    with zipfile.ZipFile(source_file, 'r') as zf:
        zipInfo = zf.infolist()
        for member in zipInfo:
            try:
                # print(member.filename.encode('cp437').decode('euc-kr', 'ignore'))
                member.filename = member.filename.encode('cp437').decode('euc-kr', 'ignore')
                zf.extract(member, dest_path)
            except:
                print(source_file)
                raise Exception('what?!')

if __name__ == "__main__":
    paths = [
        '/opt/ml/project/final-project-level3-cv-12/dataset/Training/',
        '/opt/ml/project/final-project-level3-cv-12/dataset/Validation/'
    ]
    for path in paths:
        for file in ['image.zip', 'label.zip']:
            unzip(path+file, path+file.split('.')[0])
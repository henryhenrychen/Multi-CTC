# Reference: http://homepage.ntu.edu.tw/~karchung/ChineseIPAJimmyrev.pdf
table = [
    ('ㄧㄚ',  'jɑ',  None,  'iɑ'),
    ('ㄧㄠ', 'jɑu',  None, 'iɑu'),
    ('ㄧㄝ',  'jɛ',  None,  'iɛ'),
    ('ㄧㄡ', 'jou',  None, 'iou'),
    ('ㄧㄛ',  'jɔ',  None,  None), # add
    ('ㄧㄢ', 'jɛn',  None, 'iɛn'),
    ('ㄧㄞ', 'jai',  None,  None), # add
    ('ㄧㄤ', 'jɑŋ',  None, 'iɑŋ'),
    ('ㄧㄣ', 'jin',  None,  'in'),
    ('ㄧㄥ', 'jiŋ',  None,  'iŋ'),
    ('ㄩㄝ',  'yɛ',  None,  'yɛ'),
    ('ㄩㄢ', 'yan',  None, 'yan'),
    ('ㄨㄚ',  'wɑ',  None,  'uɑ'),
    ('ㄨㄛ',  'wɔ',  None,  'ʊɔ'),
    ('ㄨㄞ', 'wai',  None, 'uai'),
    ('ㄨㄟ', 'wei',  None, 'uei'),
    ('ㄨㄢ', 'wan',  None, 'uan'),
    ('ㄨㄣ', 'wʊn',  None,  'ʊn'),
    ('ㄨㄤ', 'wɑŋ',  None, 'uɑŋ'),
    ('ㄨㄥ', 'wɔŋ',  None,  'ɔŋ'),
    ('ㄩㄣ',  'yn',  None,  'yn'),
    ('ㄩㄥ', 'jɔŋ',  None, 'iɔŋ'),
    ('ㄅ',   None,   'p', None),
    ('ㄆ',   None,  'pʰ', None),
    ('ㄇ',   None,   'm', None),
    ('ㄈ',   None,   'f', None),
    ('ㄉ',   None,   't', None),
    ('ㄊ',   None,  'tʰ', None),
    ('ㄋ',   None,   'n', None),
    ('ㄌ',   None,   'l', None),
    ('ㄍ',   None,   'k', None),
    ('ㄎ',   None,  'kʰ', None),
    ('ㄏ',   None,   'h', None),
    ('ㄐ',   None,  'tɕ', None),
    ('ㄑ',   None, 'tɕʰ', None),
    ('ㄒ',   None,   'ɕ', None),
    ('ㄓ',  'tʂɭ',  'tʂ', None),
    ('ㄔ', 'tʂʰɭ', 'tʂʰ', None),
    ('ㄕ',   'ʂɭ',   'ʂ', None),
    ('ㄖ',   'ʐɭ',   'ʐ', None),
    ('ㄗ',  'tsɨ',  'ts', None),
    ('ㄘ', 'tsʰɨ', 'tsʰ', None),
    ('ㄙ',   'sɨ',   's', None),
    ('ㄚ',    'ɑ',  None,  'ɑ'),
    ('ㄛ',    'ɔ',  None,  'ɔ'),
    ('ㄜ',    'ə',  None,  'ə'),
    ('ㄝ',   None,  None,  'ɛ'),
    ('ㄞ',   'ai',  None, 'ai'),
    ('ㄟ',   'ei',  None, 'ei'),
    ('ㄠ',   'ɑu',  None, 'ɑu'),
    ('ㄡ',   'ou',  None, 'ou'),
    ('ㄢ',   'an',  None, 'an'),
    ('ㄣ',   'ən',  None, 'ən'),
    ('ㄤ',   'ɑŋ',  None, 'ɑŋ'),
    ('ㄥ',   None,  None, 'əŋ'),
    ('ㄦ',    'ɚ',  None, None),
    ('ㄧ',    'i',  None,  'i'),
    ('ㄨ',    'w',  None,  'u'),
    ('ㄩ',    'y',  None,  'y')
]


class Converter_zh_ipa(object):
    def __init__(self, table_path):
        z2i, i2z = self._parseTable(table_path)
        self.z2i = z2i
        self.i2z = i2z

    def zh2ipa(self, zh, taiwan=True):
        if taiwan:
            # 去捲舌音
            rule1 = [
                ('ㄓ', 'tʂ', 'ts'),
                ('ㄔ', 'tʂʰ', 'tsʰ'),
                ('ㄕ', 'ʂ', 's'),
                ('ㄖ', 'ʐ', 'l')
            ]
            # 部份 'ㄥ' 換成 'ㄣ'
            rule2 = [
                'ㄉㄥ', 'ㄊㄥ', 'ㄋㄥ', 'ㄌㄥ',
                'ㄍㄥ', 'ㄎㄥ', 'ㄏㄥ',
                'ㄓㄥ', 'ㄔㄥ', 'ㄕㄥ',
                'ㄗㄥ', 'ㄘㄥ', 'ㄙㄥ',
                'ㄧㄥ', 'ㄅㄧㄥ', 'ㄆㄧㄥ', 'ㄇㄧㄥ',
                'ㄉㄧㄥ', 'ㄊㄧㄥ', 'ㄋㄧㄥ', 'ㄌㄧㄥ',
                'ㄐㄧㄥ', 'ㄑㄧㄥ', 'ㄒㄧㄥ'
            ]
            # 去 'ㄦ'
            rule3 = ('ㄦ', 'ㄜ')

            ipa = []
            for z in zh.split(' '):
                z_no_tone = z if z[-1] not in ['ˊ', 'ˇ', 'ˋ', '˙'] else z[:-1]
                tone = None if z[-1] not in ['ˊ', 'ˇ', 'ˋ', '˙'] else z[-1]
                i = self.z2i[z]
#                 print(z_no_tone, tone, i)
                # Rule 1
                for r in rule1:
                    if r[0] in z:
                        i = i.replace(r[1], r[2])
                        if len(z_no_tone) == 1:
                            i = i.replace('ɭ', 'ɨ')
                        break
                # Rule 2
                for r in rule2:
                    if z_no_tone == r:
                        assert i[-2] == 'ŋ'
                        i = i[:-2] + 'n' + i[-1]
                        break
                # Rule 3
                if z_no_tone == rule3[0]:
                    i = self.z2i[z.replace('ㄦ', 'ㄜ')]
                ipa += [i]

            return ' '.join(ipa).replace('l', 'ɫ')
        else:
            return ' '.join([self.z2i[z] for z in zh.split(' ')]).replace('l', 'ɫ')

    def ipa2zh(self, ipa):
        return [ipa2zh[i] for i in ipa]

    def _parseTable(self, table_path):
        z2i = dict()
        i2z = dict()
        with open(table_path, mode='r') as f:
            content = f.readlines()
        for line in content:
            parts = line[:-1].split(' ')
            zh, ipa = parts[0], parts[1]
            z2i[zh] = ipa
            i2z[ipa] = zh
        return z2i, i2z


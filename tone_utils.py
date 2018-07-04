CLASS_A = "a à á ả ã ạ ă ằ ắ ẳ ẵ ặ â ầ ấ ẩ ẫ ậ".split()
CLASS_E = "e è é ẻ ẽ ẹ ê ề ế ể ễ ệ".split()
CLASS_I = "i ì í ỉ ĩ ị".split()
CLASS_O = "o ò ó ỏ õ ọ ô ồ ố ổ ỗ ộ ơ ờ ớ ở ỡ ợ".split()
CLASS_U = "u ù ú ủ ũ ụ ư ừ ứ ử ữ ự".split()
CLASS_Y = "y ỳ ý ỷ ỹ ỵ".split()
CLASS_D = "d đ".split()

# 6 tones in total, including "unmarked" mark
# ngang = 0, huyền = 1, sắc = 2, hỏi = 3, ngã = 4, nặng = 5

NUM_TONES = 6
_ALL_CLASSES = [CLASS_A, CLASS_E, CLASS_I, CLASS_O, CLASS_U, CLASS_Y, CLASS_D]


def is_same_class(char1, char2):
    for c in _ALL_CLASSES:
        if char1 in c and char2 in c:
            return True
    return False


def find_tone(syllable):
    for c in syllable:
        # found a vowel
        for cls in _ALL_CLASSES:
            # it is a toned vowel
            if c in cls[1:]:
                return cls.index(c) % NUM_TONES
    # unmarked
    return 0


def change_tone(syllable, tone_type):
    if tone_type > NUM_TONES:
        return syllable
    result = []
    changed = False
    for c in syllable:
        if changed:
            result.append(c)
            continue
        # found a vowel
        for cls in _ALL_CLASSES:
            # it is a toned vowel
            if c in cls[1:] and cls.index(c) % NUM_TONES != 0:
                changed = True
                c = cls[cls.index(c) - (cls.index(c) % NUM_TONES) + tone_type]
                break
        result.append(c)
    return ''.join(result)


def clear_all_marks(syllable):
    result = []
    for c in syllable:
        for cls in _ALL_CLASSES:
            if c in cls:
                c = cls[0]
                break
        result.append(c)
    return ''.join(result)


# print(find_tone("chuyên"))
# print(find_tone("chuộn"))
# print(clear_all_marks("chọèn"))
# print(change_tone("sâu", 5))

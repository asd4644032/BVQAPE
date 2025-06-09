import re

def adjust_values(text):
    def adjust_number(match):
        full_match = match.group(0)
        content = match.group(1)
        value = float(match.group(2))
        
        if value > 1.1:
            # 將值縮放到1.0到1.1之間
            new_value = 1.0 + (value - 1.1) * (0.1 / (value - 1.0))
            new_value = round(new_value, 2)
            return f"({content}:{new_value})"
        return full_match

    # 使用正則表達式匹配括號內的內容和數值
    pattern = r'\((.*?):(\d+(\.\d+)?)\)'
    adjusted_text = re.sub(pattern, adjust_number, text)
    
    return adjusted_text

# 測試函數
test_cases = [
    "(delicate ceramic bowl:1.5)",
    "(candle:1.5)} with ((soft warm glow:0.9)) and (flickering flame:0.8) partially hidden by a ((microwave:0.7)) in a (modern kitchen setting:0.3) with (white walls:0.2), (chrome appliances:0.2), and (dark hardwood floor:0.1)",
    "(delicate ceramic bowl:0.8) sitting on the grass in front of (cute bunny rabbit:1.2) who is eating a carrot, with a bright sunny sky and fluffy white clouds behind the bunny",
    "(item1:1.8) and (item2:1.3) but (item3:0.9)"
]

for case in test_cases:
    print(f"Original: {case}")
    print(f"Adjusted: {adjust_values(case)}")
    print()

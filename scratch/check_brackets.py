import re

def check_brackets(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the script block
    match = re.search(r'<script type="text/babel">(.*?)</script>', content, re.DOTALL)
    if not match:
        print("No babel script found")
        return
    
    code = match.group(1)
    
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    lines = code.split('\n')
    
    for r_num, line in enumerate(lines, 1):
        # Ignore comments and strings to make checking more robust (basic check)
        # We'll just do a character check for now but keep track of line numbers
        for c_num, char in enumerate(line, 1):
            if char in '({[':
                stack.append((char, r_num, c_num))
            elif char in ')}]':
                if not stack:
                    print(f"Extra closing {char} at line {r_num}, col {c_num}")
                    return
                top, l, c = stack.pop()
                if top != mapping[char]:
                    print(f"Mismatched {char} at line {r_num}, col {c_num} (matches {top} at line {l}, col {c})")
                    return
                    
    if stack:
        print(f"Unclosed brackets/braces/parentheses left: {len(stack)}")
        for item in stack[:5]:
            print(f"  Unclosed {item[0]} at line {item[1]}, col {item[2]}")
    else:
        print("Brackets/braces/parentheses match perfectly!")

if __name__ == '__main__':
    check_brackets('/media/brammesh/New Volume/Final year project/Project/Heart_rate/static/index.html')

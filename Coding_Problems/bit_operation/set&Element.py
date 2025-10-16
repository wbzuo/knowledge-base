# 空集
0
# 单元素集合 {i}
i = 3
1 <<  i # 2^3

# 全集 {0, 1, 2, 3, ..., n - 1}
i = n = 10
(1 << n) - 1

# 补集
# ((1 << n) - 1) ⊕ s

# 属于
# (s >> i) & 1==1
# # 不属于
# (s >> i) & 1==0

# 添加元素
# s ∣ (1 << i)

# 删除元素
# s&∼(1 << i)
# 删除最小元素
# s&(s−1)


# 遍历集合

s = 15 # 1111

for i in range(4):
    if (s >> i) & 1:
        print(1)
        
        

t = s
while t:
    lowbit = t & -t
    t ^= lowbit
    i = lowbit.bit_length() - 1
    # 处理 i 的逻辑
    


# 枚举集合 
for s in range(1 << n):
    # 处理每个集合
    pass

# 枚举非空子集
# ub = s
while sub:
    # 处理 sub 的逻辑
    sub = (sub - 1) & s

# 枚举子集
# sub = s
# while True:
#     # 处理 sub 的逻辑
#     if sub == 0:
#         break
#     sub = (sub - 1) & s


# 超集
s = t
while s < (1 << n):
    # 处理 s 的逻辑
    s = (s + 1) | t

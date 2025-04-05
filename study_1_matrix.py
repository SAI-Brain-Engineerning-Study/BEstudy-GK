# 행렬 공부
# Mmn | m-행(x) n-열(y)

#정방행렬 - m=n Mn
a = [[1,2],[3,4]]
b = [[5,6],[7,8]]

#영행렬 - 모든 성분 0
print("영행렬 :", [[0,0],[0,0]])

#항등행렬 - 행렬곱 시 항등원 *대각성분(Mnn)=1
M = []
o = [[1,0],[0,1]]
for i in range(2) :
    m = []
    for j in range(2) :
        c = 0
        for k in range(2) :
            c += a[i][k]*o[k][j]
        m.append(c)
    M.append(m)
print("항등행렬 :", M)

#삼각행렬 - 대각성분 상하에 0 있는 경우

#전치행렬 - Mmn > Mnm
A = []
for i in range(2) :
    M = []
    for j in range(2) :
        c = a[j][i]
        M.append(c)
    A.append(M)
print("전치행렬 :",A)

#행렬곱(M2)
#Mmk과 Mkn의 곱.
a = [[1,2],[3,4]]
b = [[5,6],[7,8]]
M = []
for i in range(2) :
    m = []
    for j in range(2) :
        c = 0
        for k in range(2) :
            c += a[i][k]*b[k][j]
        m.append(c)
    M.append(m)
print("행렬곱 :",M)


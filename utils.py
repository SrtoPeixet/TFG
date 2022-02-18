import os
dir = os.getcwd() + '\img'
print(dir)
cnt = 0
for x in os.walk(dir):
    if '__' in x[0]:
        print(x[0])
        cnt +=1

print(cnt)

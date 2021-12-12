weight = 0.5 # 첫번째는 임의 또는 랜덤하게 주는거다. 
x0 = 0.5 # 초기치 input
goal_prediction = 0.8 # 이놈 빼고 다 튜님된다!!
lr = 0.01 #0.1 #0.001 #0.1 / 1/ 0.001 / 100
epochs = 300

for iteration in range(epochs):
    print('weight : ', weight)
    prediction = x0 * weight
    error = ( prediction - goal_prediction) ** 2

    print("Error : " + str(error) + "\tPrediction : " + str(prediction))
    # print('\n')

    up_prediction = x0 * (weight + lr)
    up_error = (goal_prediction - up_prediction) ** 2

    down_prediction = x0 * (weight - lr)
    down_error = (goal_prediction - down_prediction) ** 2

    if(down_error < up_error) :
        weight = weight - lr
    if(down_error > up_error):
        weight = weight + lr

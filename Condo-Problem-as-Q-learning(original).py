import numpy as np
import time as tic
import xlwt as xlw

t = tic.time()

# Macros

prob_lambda = 0.9
number_periods = 10
quality = [3, 2, 1, 0]
init_state = [2, 2, 2]
types = len(init_state)
end_state = [0, 0]

subdivisions = 6
increment = []
for i in range(types):
    increment.append(quality[i]/subdivisions) # Assume quality = max(price) for a given item

possible_prices = []

# Setting Constraints: first parameter is time, second sales, third revenue, fourth penalty

con = [[1,1,0,25],[6,3,0,25]]

#Assembling the matrix of possible prices
for i in range(types):
    price_type = []
    
    add_price = 0
    for j in range(subdivisions + 1):
        price_type.append(add_price)
        add_price += increment[i]
    possible_prices.append(price_type)

# R function, computes revenue naively as the product of price and sales

def R(state, price, new_state):
    R = 0
    for i in range(len(state)):
        R += price[i] * (state[i] - new_state[i])    
    return R

# Q matrix

shape = init_state[0:types]
for i in range(len(shape)):
    shape[i] += 1
    
print(shape)

shape.append(number_periods)
shape.append((subdivisions + 1) ** types)
Q = np.zeros(shape)

# Parameters (learning rate, exploration rate, discount factor) 

gamma = 0.9
alpha = 0.001
epsilon = [0.5, 0.99999, 0.25]

# This function returns all available actions in the state

def available_actions():
    av_act = []
    for i in range((subdivisions + 1) ** types):
        av_act.append(i)
    return av_act
    
# Checks for constraints    
def constraints(state, revenue):
    # for i in range(len(con)):
    #     if time == con[i][0]:
    #         penalty = con[i][3]
    #         sales_volume = 0
    #         
    #         for j in range(types):
    #             sales_volume += (init_state[j] - state[j])
    #             
    #         if sales_volume < con[i][1] or revenue < con[i][2]:
    #             return penalty                
    #         else:
    #             return 0    

    return 0
                

# Single Timestep for Customer Arrival/Purchase
mean = 5
std_dev = 2

def transition(state, price):
    prob_arrive = (np.random.normal(mean/(time+1),std_dev/(time+1)))
    if prob_arrive <= 0:
        prob_arrive = 0

    prob_arrive = round(prob_arrive)

    for i in range(prob_arrive):
        prob_purchase = []
        theta = np.random.lognormal(-1,0.5,1)[0]
        for i in range(types):
            if state[i] != 0:
                prob_purchase.append(theta * quality[i] - price[i])

            else:
                prob_purchase.append(-10000)

        prob_purchase.append(0)

        counter = prob_purchase.count(max(prob_purchase))

        if counter > 1:
            best_purchase_mult = []
            for i in range(type + 1):
                if prob_purchase[i] == max(prob_purchase):
                    best_purchase_mult.append(i)
            best_purchase = np.random.choice(best_purchase_mult,1)

        elif counter == 1:
            best_purchase = prob_purchase.index(max(prob_purchase))

        if best_purchase == types:
            return

        else:
            state[best_purchase] += -1
    
    # prob_purchase = []
    # prob_arrive = np.random.uniform(0,1,1)[0]
    # theta = np.random.uniform(0,1,1)[0]        
    #  
    # if prob_arrive <= prob_lambda:       
    #     for i in range(types):
    #         if state[i] != 0:
    #             prob_purchase.append(theta * quality[i] - price[i])
    # 
    #         else:
    #             prob_purchase.append(-10000)

   ##   else:
    #     return

   ##   prob_purchase.append(0)

   ##   counter = prob_purchase.count(max(prob_purchase))

   ##   if counter > 1:
    #     best_purchase_mult = []
    #     for i in range(type + 1):
    #         if prob_purchase[i] == max(prob_purchase):
    #             best_purchase_mult.append(i)
    #     best_purchase = np.random.choice(best_purchase_mult,1)

   ##   elif counter == 1:
    #     best_purchase = prob_purchase.index(max(prob_purchase))

   ##   if best_purchase == types:
    #     return

   ##   else:
    #     state[best_purchase] += -1

   ##   return

# This function chooses at random which action to be performed within the range 
# of all the available actions.

def sample_next_action(available_actions):
    
    next_action = np.random.choice(available_actions,1)
    return next_action

# One timestep for Q-learning
def single_time_step():
    global time, state, revenue, convergence_immediate
    
    #Getting action based on epsilon-exploration coefficient
    random = np.random.uniform(0,1)

    Q_value = np.copy(Q)
    
    for i in range(types):
        Q_value = Q_value[state[i]]

    Q_value = Q_value[time]
    
    if random >= epsilon[0]:
        max_Q = np.amax(Q_value)
        L = []
        for i in range(len(Q_value)):
            if Q_value[i] == max_Q:
                L.append(i)
        price_index = sample_next_action(L)[0]
        
    else:
        av_act = available_actions()
        price_index = sample_next_action(av_act)[0]
        
    price = []
    for i in range(types - 1):
        price.append(possible_prices[i][price_index // ((subdivisions + 1) ** (types - i - 1))])
        price_index = price_index % (subdivisions + 1)
        
    price.append(possible_prices[types - 1][price_index])

    temp_state = np.copy(state)
    transition(state, price)
    revenue += R(temp_state, price, state)  
    
    #Updating Q-values
    next_Q_value = np.copy(Q)    
    for i in range(types):
        next_Q_value = next_Q_value[state[i]]        
    next_Q_value = next_Q_value[time]
    argmax = np.amax(next_Q_value)
        
    reward = R(temp_state, price, state) - constraints(state,revenue)
    Q_old = np.copy(Q)    
    for i in range(types):
        Q_old = Q_old[temp_state[i]]        
    Q_old = Q_old[time, price_index]

    Q[temp_state[0], temp_state[1], temp_state[2], time, price_index] += alpha * (reward + gamma * argmax - Q_old)
    
    convergence_immediate += abs(alpha * (reward + gamma * argmax - Q_old))
    
    time += 1
    
    return
    
# For testing purposes    
def single_time_step_test(Q):
    global time, state, revenue
    
    Q_value = np.copy(Q)
    
    for i in range(types):
        Q_value = Q_value[state[i]]

    Q_value = Q_value[time]   
 
    max_Q = np.amax(Q_value)
    L = []
    for i in range(len(Q_value)):
        if Q_value[i] == max_Q:
            L.append(i)
    price_index = sample_next_action(L)[0]

    price = []
    for i in range(types - 1):
        price.append(possible_prices[i][price_index // ((subdivisions + 1) ** (types - i - 1))])
        price_index = price_index % (subdivisions + 1)
        
    price.append(possible_prices[types - 1][price_index])
      
    temp_state = np.copy(state)
    transition(state, price)
    revenue += R(temp_state, price, state)  
    
    time += 1    
    
    return
    
# Q-learning and Output 
convergence_list = []
increment = 1000
repetitions = 1

for i in range(50000):
    state = []
    for j in range(types):
        state.append(init_state[j])
    time = 0
    revenue = 0
    convergence_immediate = 0
    
    while(time != number_periods and state != end_state):
        single_time_step()
    
    if epsilon[0] > epsilon[2]:
        epsilon[0] *= epsilon[1]
        
    if i%increment == 0:
        print(i, tic.time() - t)
        
        convergence_list.append(convergence_immediate) 
        # Q_temp = np.copy(Q)
        # 
        # expected = 0
       
        #   for j in range(repetitions):
        #     state = []
        #     for k in range(types):
        #         state.append(init_state[k])
        #     time = 0
        #     revenue = 0
        #     
        #     while(time != number_periods and state != end_state):
        #         single_time_step_test(Q_temp)
        # 
        #     expected += revenue
        #     
        # expected = expected/repetitions
        # convergence_list.append(expected)
        
    if i == 49999:
        print(revenue)
        
book = xlw.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet 1")
sheet1.write(0,0,"Iteration Number")
sheet1.write(0,1,"Expected Revenue")
# sheet1.write(0,2,"Theta Min")
# sheet1.write(0,3,"Theta Max")
# 
for i in range(len(convergence_list)):
    sheet1.write(i+1, 0, i * increment)
    sheet1.write(i+1, 1, convergence_list[i])
#     sheet1.write(i+1, 2, good_values[i][2])
#     sheet1.write(i+1, 3, good_values[i][3])

book.save("Q convergence 3-8-10-50k.xls")        

#         
# # Printing Best Prices
# for time in range(number_periods):
#     print("At time t =", time)
#     for i in range(init_state[0] + 1):
#         for j in range(init_state[1] + 1):
#             Q_value = Q[i,j,time]
#             k = np.argmax(Q_value)
#             if [i,j] == [0,0]:
#                 continue
#             elif j == 0:
#                 print("When there are", i, "units of A remaining", "Best price is:", possible_prices[0][k // (subdivisions + 1)], "for A.")
#             elif i == 0:
#                 print("When there are", j, "units of B remaining", "Best price is:", possible_prices[1][k % (subdivisions + 1)], "for B.")
#             else:
#                 print("When there are", i, "units of A remaining,", j, "units of B remaining", "Best prices are:", possible_prices[0][k // (subdivisions + 1)], "for A", possible_prices[1][k % (subdivisions + 1)], "for B.")
#     print("\n")

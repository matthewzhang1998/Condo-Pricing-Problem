import numpy as np
import time as tic

t = tic.time()

# Macros

prob_lambda = 0.9
number_periods = 8
quality = [13, 9, 0]
init_state = [3, 3]
types = len(init_state)
end_state = [0, 0]

subdivisions = 10
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
shape.append(number_periods)
shape.append((subdivisions + 1) ** types)
Q = np.zeros(shape)

# Parameters (learning rate, exploration rate, discount factor) 

gamma = 0.9
alpha = 0.1
epsilon = [0.5, 0.999, 0.25]

# This function returns all available actions in the state

def available_actions():
    av_act = []
    for i in range((subdivisions + 1) ** types):
        av_act.append(i)
    return av_act
    
# Checks for constraints    
def constraints(state, revenue):
    for i in range(len(con)):
        if time == con[i][0]:
            penalty = con[i][3]
            sales_volume = 0
            
            for j in range(types):
                sales_volume += (init_state[j] - state[j])
                
            if sales_volume < con[i][1] or revenue < con[i][2]:
                return penalty                
            else:
                return 0    

    return 0
                

# Single Timestep for Customer Arrival/Purchase
def transition(state, price):
   
    prob_arrive = (np.random.normal(25/time,12.5/time))
    if prob_arrive <= 0:
        prob_arrive = 0


    prob_arrive = round(prob_arrive)

    prob_purchase = []

    for i in range(prob_arrive):
        theta = np.random.lognormal(-1,0.5,1)
        for i in range(types):
            if state[i] != 0:
                prob_purchase.append(theta * quality[i] - price[i])

                else:
                    prob_purchase.append(-10000)
        else:
            return

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

        return
# This function chooses at random which action to be performed within the range 
# of all the available actions.

def sample_next_action(available_actions):
    
    next_action = np.random.choice(available_actions,1)
    return next_action

# One timestep for Q-learning
def single_time_step():
    global time, state, revenue
    
    #Getting action based on epsilon-exploration coefficient
    random = np.random.uniform(0,1)
    Q_value = Q[state[0],state[1],time,]
    
    if random >= epsilon[0]:        
        max_Q = np.amax(Q_value)
        L = []
        for i in range(len(Q_value)):
            if Q_value[i] == max_Q:
                L.append(i)
        price_index = sample_next_action(L)[0]
        
        #Applying constraint (hard coded for two categories; need to rewrite for general form)
        while(possible_prices[0][price_index // (subdivisions + 1)] < possible_prices[1][price_index % (subdivisions + 1)]):
            price_index = sample_next_action(L)[0]
        price = [possible_prices[0][price_index // (subdivisions + 1)], possible_prices[1][price_index % (subdivisions + 1)], 0]
        
    else:
        av_act = available_actions()
        price_index = sample_next_action(av_act)[0]
        
        #Applying similar constraint        
        while(possible_prices[0][price_index // (subdivisions + 1)] < possible_prices[1][price_index % (subdivisions + 1)]):
            price_index = sample_next_action(av_act)[0]
        price = [possible_prices[0][price_index // (subdivisions + 1)], possible_prices[1][price_index % (subdivisions + 1)], 0]

    temp_state = np.copy(state)
    transition(state, price)
    revenue += R(temp_state, price, state)  
    
    #Updating Q-values
    next_Q_values = Q[state[0],state[1],time]
    argmax = np.amax(next_Q_values)
        
    reward = revenue - constraints(state,revenue)
    Q_old = Q[temp_state[0], temp_state[1], time, price_index]
    Q[temp_state[0], temp_state[1], time, price_index] += alpha * (reward + gamma * argmax - Q_old)
    
    time += 1
    
    return
    
# Q-learning and Output 
for i in range(50000):
    state = []
    for j in range(types):
        state.append(init_state[j])
    time = 0
    revenue = 0
    
    while(time != number_periods and state != end_state):
        single_time_step()
    
    if epsilon[0] > epsilon[2]:
        epsilon[0] *= epsilon[1]
        
    if i%1000 == 0:
        print(i, tic.time() - t)
        
    if i == 49999:
        print(revenue)
        
# Printing Best Prices
for time in range(number_periods):
    print("At time t =", time)
    for i in range(init_state[0] + 1):
        for j in range(init_state[1] + 1):
            Q_value = Q[i,j,time]
            k = np.argmax(Q_value)
            if [i,j] == [0,0]:
                continue
            elif j == 0:
                print("When there are", i, "units of A remaining", "Best price is:", possible_prices[0][k // (subdivisions + 1)], "for A.")
            elif i == 0:
                print("When there are", j, "units of B remaining", "Best price is:", possible_prices[1][k % (subdivisions + 1)], "for B.")
            else:
                print("When there are", i, "units of A remaining,", j, "units of B remaining", "Best prices are:", possible_prices[0][k // (subdivisions + 1)], "for A", possible_prices[1][k % (subdivisions + 1)], "for B.")
    print("\n")

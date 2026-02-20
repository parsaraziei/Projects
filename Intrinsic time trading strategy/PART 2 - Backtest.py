import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
folder_path = "C:\\Users\\parsa\\OneDrive\\Desktop\\exchange_data\\AUS-USD"
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
num_files = len(files)
#used to determine the number of files in a folder

back_test = pd.DataFrame(columns=["num","positive-trades","negative-trades","profit","win-ratio"])


# to open a long position
def long_position():
    global long_active, open_price_long, coef, threshold_factor, SL_long, TP_long  # Declare the variables as global
    if long_active == False:  # only allowed if there are no other long positions active
        long_active = True  # position is active now
        open_price_long = IE_points.iloc[-1]  # this will fetch the most recent directional change, which we are currently at, and store it in opening price
        print("long at "+ str(open_price_long["ask"])) # print how much we bought the asset for now

        TP_long = (1+coef*threshold_factor)*open_price_long["ask"] # take profit is calculated using the formula above
        SL_long = (1-coef*threshold_factor)*open_price_long["ask"] # stop loss is calculated using the formula above
 
        length = len(strategy_overview) #gets the number of positions we have had so far
        strategy_overview.loc[length] = open_price_long # the current timestep which is a directional is added to the positions opened
        strategy_overview.loc[length,"action"] = "open long" # the type of the position is set to long
        strategy_overview.loc[length,"type"] = "DC UP" # the position is opened when we have an upwards directional change
        

# to close a long position
def close_long(current_timestep):
    global long_active, open_price_long, close_price_long, loss_trades, profitable_trades, profit  # Declare the variables as global
    if long_active: # only close long if there is one active
        long_active = False # deactivate the position, i.e cashout
        close_price_long = current_timestep #closing price is the current price of the exchanged pair
        trade_profit = (close_price_long["bid"] - open_price_long["ask"]) #the profit is the difference between the ask now 
        # and bid when position was opened
        profit += trade_profit
        # this trades profit is added to overall profit
        print("close long at " + str(close_price_long["bid"]) + " profit: "+ str(trade_profit))
        
        length = len(strategy_overview)
        strategy_overview.loc[length] = close_price_long
        strategy_overview.loc[length,"date-time"] = close_price_long.name
        strategy_overview.loc[length,"action"] = "close long"
        strategy_overview.loc[length,"type"] = str(trade_profit)
        #the details of the current timestep in which we took the action of closing this long position and the profit this 
        #postion has made is stored in the dataframe of all the positions so far.

        if(trade_profit<=0):
            loss_trades += 1
            #if lost money, number of bad trades gets incremented by 1
        else:
            profitable_trades += 1
            #if made money, number of profiable trades gets incremented by 1
    
#to open a short position
def short_position():
    global short_active, profit, open_price_short, close_price_short , coef, SL_short, TP_short, threshold_factor
    if(short_active == False):
        short_active = True # only allowed if there are no other short positions active
        open_price_short = IE_points.iloc[-1]  # this will fetch the most recent directional change, which we are currently at, stores it in opening-price
        print("short at "+ str(open_price_short["bid"]))

        SL_short = open_price_short["bid"]*(1 + coef * threshold_factor) # compute Stop Loss for this position using the formula
        TP_short = open_price_short["bid"]*(1 - coef * threshold_factor) # computes Take profit for this position using the formula
        
        length = len(strategy_overview)#gets the number of positions we have had so far
        strategy_overview.loc[length] = open_price_short # the current timestep which is a directional is added to the positions opened
        strategy_overview.loc[length,"action"] = "open short" # the type of the position is set to short
        strategy_overview.loc[length,"type"] = "DC DOWN" # the position is opened when we have an downwards directional change        

# to close a short position
def close_short(current_timestep):
    global short_active, profit, open_price_short, close_price_short,  loss_trades, profitable_trades, profit  # Declare the variables as global
    if short_active: # only close a short if there is one active
        short_active = False # then set the existence of an active short position to false
        close_price_short = current_timestep # the current timestep is when we close the short position
        trade_profit = (open_price_short["bid"] - close_price_short["ask"]) # the profit is calculated by the open positions ask subtracted from closes bid
        profit += trade_profit # the profit of this trade is added to the net profit so far
        
        print("close short at " + str(close_price_short["ask"]) + " profit: "+ str(trade_profit))
        
        length = len(strategy_overview)
        strategy_overview.loc[length] = close_price_short
        strategy_overview.loc[length,"date-time"] = close_price_short.name
        strategy_overview.loc[length,"action"] = "close short"
        strategy_overview.loc[length,"type"] = str(trade_profit)
        # the details of the current timestep in which we took the action of closing this short position and the profit this 
        # postion has made is stored in the dataframe of all the positions we have opened so far.
        if(trade_profit<0):
            loss_trades += 1
            #if lost money, number of bad trades gets incremented by 1
        else:
            profitable_trades += 1
            #if made money, number of profiable trades gets incremented by 1
        
      

# if the price goes below the price point mentioned above, the adjust short is called
# This method is used to change the SL and TP of a short position
def adjust_short(current_timestep):
    global SL_short,TP_short, open_price_short,threshold_factor
    if short_active: # only when there is a short position active
        """uncomment the print statements to see the changes to the TP and SL"""
        #print("current SL:"+str(SL_short) + " and TP : "+ str(TP_short)) # the previous TP and SL for the short position
        SL_short = current_timestep["bid"]*(1+(19/20)*coef*threshold_factor) # calculate the new SL for the short positon
        TP_short = current_timestep["bid"]*(1-(1/20)*coef*threshold_factor) # calculate the new TP for the short positon
        #print("changed  SL:"+str(SL_short) + " and TP : "+ str(TP_short)) # the new TP and SL for the short position


# if the price goes above the price point mentioned above, the adjust long is called
# This method is used to change the SL and TP of a short position
def adjust_long(current_timestep):
    global SL_long,TP_long, open_price_long,threshold_factor
    if long_active: # only when there is a short position active
        """uncomment the print statements to see the changes to the TP and SL"""
        #print("current SL:"+str(SL_long) + " and TP : "+ str(TP_long)) # the previous TP and SL for the long position
        SL_long = current_timestep["ask"]*(1-(19/20)*coef*threshold_factor)  # calculate the new SL for the long positon
        TP_long = current_timestep["ask"]*(1+(1/20)*coef*threshold_factor)  # calculate the new TP for the long positon
        #print("changed  SL:"+str(SL_long) + " and TP : "+ str(TP_long)) # the new TP and SL for the long position
    



for month in range(1,num_files+1):
    df = pd.read_csv("C:\\Users\\parsa\\OneDrive\\Desktop\\exchange_data\\AUS-USD\\"+str(month)+".csv", names=["bid", "ask", "label"])
    
    df.index = pd.to_datetime(df.index, format = "%Y%m%d %H%M%S%f")
    

    if((df["label"]!=0).sum()==0):
        del df["label"]
    
    df.head()

    df["mid"] = (df["bid"]+df["ask"])/2
    tickdata = df.sort_index(ascending=True)
    
    """ defining the global variables to use later in trading"""
    short_active = False; long_active = False
    # initially, no positions are open
    profit = 0
    # profit is initially 0
    open_price_short = None; open_price_long = None
    close_price_short = None; close_price_long = None
    # the opening and closing prices for our trades
    profitable_trades = 0
    # number of trades that resulted in profit 
    loss_trades = 0
    # number of trades that resulted in loss 
    coef = 0.7

    SL_short = None ; TP_short = None
    # The stoploss and take profit for short positions (changes when a new short is opened)
    SL_long = None ; TP_long = None
    # The stoploss and take profit for long positions (changes when a new long is opened)

    strategy_overview = pd.DataFrame(columns=["date-time","bid", "ask", "mid", "type","action"])
    

    # The first point in other words to draw the analogy the price at this moment is set as the local extreme. 
    # (since we have not encountered future prices yet)
    min_ext = max_ext = tickdata.iloc[0]
    # unsure about the first direction of the price
    IE_points = pd.DataFrame(columns=["date-time","bid", "ask", "mid", "type"])
    # Dataframe created to store the local extremes and directional change points 
    counter = 0; 
    # used to iterate through the data
    assigner = 0
    # used as the 1 + final index in the IE_points, used to add new points to intrinsic time.
    threshold_factor = 0.001
    upwards_movement = None
    # if true means moving upwards, if false means downwards trend, currently it's neither.

    # while loop to only determine the first directional change, whether it is an upwards trend or a downwards.
    while (len(IE_points) == 0 and counter < len(tickdata)):
        # we search till we find the first intrinsic point i.e, or run out of the whole month's data, 
        # (no DCs the whole month, which is extremely unlikely) for this value of theta
        current_tick = tickdata.iloc[counter]
        # current tick is the tick data at the index of the counter

        # If the price is higher than the limit we set, then an upwards DC is detected
        if(min_ext["mid"] * (threshold_factor + 1) < current_tick["mid"]):
            
            IE_points.loc[assigner] = min_ext
            IE_points.loc[assigner,"type"] = "EXT"
            IE_points.loc[assigner,"date-time"] = min_ext.name
            # the local extreme, minimum in this case and the quotes at its timestep
            # are stored in the data from with the type of EXT
            assigner+=1
            # The index is incremented
            
            IE_points.loc[assigner] = current_tick
            IE_points.loc[assigner, "type"] = "DC"
            IE_points.loc[assigner,"date-time"] = current_tick.name
            # The current timestep and its details whose value exceeded the limit for
            # Upwards movement is stored in the dataframe 
            assigner += 1
            # increment to the next available spot, since this one is assigned
            upwards_movement = True
            # The direction of the price trend is now set to upwards.
            max_ext = current_tick
            # Now the local extreme is the current_tick, and we start searching for a new maximum. 
            long_position()
            # open a long position, since the direction is going upwards.
            

        # If the current price is lower than the limit we set, then a downward DC is detected
        elif (max_ext["mid"] * (1 - threshold_factor) > current_tick["mid"]):
            IE_points.loc[assigner] = max_ext
            IE_points.loc[assigner, "type"] = "EXT"
            IE_points.loc[assigner,"date-time"] = max_ext.name
            # the local extreme, maximum in this case and the quotes at its timestep
            # are stored in the data from with the type of EXT
            assigner += 1
            # The index is incremented
            
            IE_points.loc[assigner] = current_tick
            IE_points.loc[assigner, "type"] = "DC"
            IE_points.loc[assigner,"date-time"] = current_tick.name
            # The current timestep and its details, whose value drops below the limit for
            # downwards movement is stored in the dataframe 
            assigner += 1
            # increment to the next available spot, since this one is assigned
            upwards_movement = False
            # The direction of the price trend is now set to downwards.
            min_ext = current_tick
            # Now the local extreme is the current_tick, and we start searching for a new minimum. 
            #short_position()
            # open a short position, since the direction is going downwards.
        
        else:    
            if (current_tick["mid"]>max_ext["mid"]): max_ext = current_tick
            # If the current price is higher than the maximum we had, this would be the new maximum
            elif (current_tick["mid"]<min_ext["mid"]): min_ext = current_tick
            # If the current price is lower than the minimum we had, this would be the new minimum
        counter += 1
        # The counter which is used to iterate through the tick data is incremented by one.

    tickdata = tickdata[current_tick.name:]
# We only look at the tick data from the current directional change onwards, since we have already found the directional change in this period.
    current_DC = IE_points.iloc[-1]
# The current directional change variable is set to the last one (DC) discovered 

# A for loop to go through all the tick data 
    for timestep in range(0,len(tickdata)):
        current_tick = tickdata.iloc[timestep]
        # the current tick is the new timestep.
        if upwards_movement:
            # only look for the downwards directional change if the direction of the price is currently upwards.
            if max_ext["mid"]*(1-threshold_factor)>current_tick["mid"]:
                IE_points.loc[assigner] = max_ext
                IE_points.loc[assigner, "type"] = "EXT"
                IE_points.loc[assigner,"date-time"] = max_ext.name
                # the local extreme, maximum in this case and the quotes at its timestep
                # are stored in the dataframe with the type of EXT
                assigner += 1
                # The index is incremented
            
                IE_points.loc[assigner] = current_tick
                IE_points.loc[assigner, "type"] = "DC"
                IE_points.loc[assigner,"date-time"] = current_tick.name
                current_DC = IE_points.loc[assigner]
                # The current timestep and its details, whose value drops below the limit for
                # downwards movement is stored in the dataframe 
                assigner += 1
                upwards_movement = False
                # The direction of the price trend is now set to downwards.
                min_ext = current_tick
                # Now the local extreme is the current_tick, and we start searching for a new minimum.
                #short_position()
                # open a short position, since the direction is going downwards.
            else:
            # If the direction is upwards, and the current price is larger than the maximum
                if(max_ext["mid"]<current_tick["mid"]):
                    max_ext = current_tick # Then this would become the new maximum.
                    if(IE_points.iloc[-1]["mid"]*(1+threshold_factor)<max_ext["mid"]):
                        # finding further intrinsic points if the price goes higher than a certain amount upwards, they have the type further directional change
                        IE_points.loc[assigner] = max_ext
                        IE_points.loc[assigner,"date-time"] = max_ext.name
                        IE_points.loc[assigner,"type"] = "F-DC"
                        current_DC = IE_points.loc[assigner]
                        # This point and its timestamp plus the quotes are stored as the further directional change in the intrinsic points dataframe.
                        long_position() # open a long position
                        #further extreme(another max)
                        assigner+=1
        
            if long_active: # if we confirmed the trend that the price is going up, and we have an active long:
                if current_tick["bid"] <= SL_long or  current_tick["bid"] >= TP_long: 
                    #if the current bid price is smaller than the stop loss and larger than the take profit, close the long position 
                    close_long(current_tick)
                """COMMENT THIS SECTION OUT IF YOU WANT TO RUN THE BASE SHORTING AND LONGING STRATEGY MINUS (NO DYNAMIC SL AND TP CHANGES)"""
                # if (current_tick["bid"] >=  (1-coef*(1/2)*threshold_factor) * TP_long):  # if the current bid is lorger than 
                #     adjust_long(current_tick)                                            # the calculated limit to readjust the long positions SL and TP.
                """--------------------------------------------------------------------------------------------------------------------------"""   
            
                            
                
        elif not upwards_movement:  # only look for the upwards directional change if the direction of the price is currently downwards.
            if(min_ext["mid"] * (threshold_factor + 1) < current_tick["mid"]):
                IE_points.loc[assigner] = min_ext
                IE_points.loc[assigner,"type"] = "EXT"
                IE_points.loc[assigner,"date-time"] = min_ext.name
                assigner+=1
                # the local extreme, minimum in this case and the quotes at its timestep
                # are stored in the data frame with the type of EXT
                # The new position to add intrinsic points is incremented
                
                IE_points.loc[assigner] = current_tick
                IE_points.loc[assigner, "type"] = "DC"
                IE_points.loc[assigner,"date-time"] = current_tick.name
                current_DC = IE_points.loc[assigner]
                assigner += 1
                # The current timestep and its details, whose value drops below the limit for
                # Upwards movement is stored in the dataframe 
                upwards_movement = True
                max_ext = current_tick
                # Now the local extreme is the current_tick, and we start searching for a new maximum.
                long_position()
                # open a long position, since the direction is going upwards.
            else:
                if(min_ext["mid"]>current_tick["mid"]):
                #if the direction is downwards and the current tick value is smaller than the minimum tick so far, assign the new one as the local extreme
                    min_ext = current_tick
                    if(IE_points.iloc[-1]["mid"]*(1-threshold_factor)>min_ext["mid"]):
                        IE_points.loc[assigner] = min_ext
                        IE_points.loc[assigner,"date-time"] = min_ext.name
                        IE_points.loc[assigner,"type"] = "F-DC"
                        # if the current price goes below a certain amount lower than the previous downwards directional change point
                        # save this as a further directional change alongside its time and information in the intrinsic events data frame.
                        #short_position() # open a short position
                        assigner+=1
                    #increment it to be set to the next available spot
                    
            
            if short_active: 
                # If we currently have a short position active, and the 
                # price goes below TP or lower than the SL, close short
                if (current_tick["ask"] <= TP_short or current_tick["ask"] >= SL_short):
                    #close_short(current_tick)
                    """COMMENT THIS SECTION OUT IF YOU WANT TO RUN THE BASE SHORTING AND LONGING STRATEGY MINUS (NO DYNAMIC SL AND TP CHANGES)"""
                    
                # if(current_tick["ask"]<= (1+coef*(1/2)*threshold_factor) * TP_short): # if the current bid is smaller than 
                #     adjust_short(current_tick)     # the calculated limit to readjust the short positions SL and TP.                
                    """---------------------------------------------------------------------------------------------------------------------"""    
                
        
    #I still need to add the double in one direction
    IE_points.head()
    IE_TimeSeries = IE_points.groupby(["date-time", "bid", "ask", "mid"], as_index=False).agg({
    "type": " ".join
    }).reset_index(drop=True)
    # merging the two entries with the same values and concatenating their types.
    IE_TimeSeries.to_excel("C:\\Users\\parsa\\OneDrive\\Desktop\\output-refined1.xlsx")
    # The cleaned-up timeseries is stored in a desktop file
    strategy_overview.to_excel("C:\\Users\\parsa\\OneDrive\\Desktop\\positions1.xlsx")
    # All the positions and their details regarding the opening and closing is also stored in a separate file.

    print("successful positions: "+ str(profitable_trades) + ", " + "failed postions: " + str(loss_trades) +
    ", net profit:" + str(profit) + ", the win position ratio :" + str(profitable_trades/(loss_trades+profitable_trades)))

       
    back_test.loc[month] = [str(month)+"-2024",profitable_trades,loss_trades,profit,(profitable_trades/(profitable_trades+loss_trades))]
    
back_test.to_excel("C:\\Users\\parsa\\OneDrive\\Desktop\\backtst.xlsx")
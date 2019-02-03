//+------------------------------------------------------------------+
//|                                           fetcher_trader_v10.mq5 |
//|                                                       mgutierrez |
//|                                       https://www.caissandra.com |
//+------------------------------------------------------------------+
#property copyright "mgutierrez"
#property link      "https://www.caissandra.com"
#property version   "1.00"
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\OrderInfo.mqh>

#define EXPERT_MAGIC 123456   // MagicNumber of the expert

// Define variables
int const nBuffers = 100;
int const bufferSize = 10;
double buffer[10][2];// buffer containing last bids/asks
int file_index;
int bids_counter;
int filehandle;

datetime currTime;
datetime timeSoll;
datetime tic;
datetime toc;
double diffTime;
double bid;
double ask;
double Bi;
double Ai=1;
double Bi_soll;
double Ai_soll;
double Bo_soll;
double Ao_soll;

string filename;
string thisSymbol;
string directoryNameLive;
string directoryNameRecordings;
string chunks[];

// Define variables
CTrade m_Trade;
CPositionInfo m_Position;
input double Lot = 0.01;
double lot;
const double lot_in_eur = 100000;
bool record_data = true;

const int nNets = 1;
int deadline = -1;
int filehandle_trader;
int nEventsPerStat;
string predictionString;
int prediction;
string TTfile;
string LCfile;
bool timer_bool;
//datetime currTime;

double stoploss;
double B_sl;
double A_sl;
double sl_thr = 0.0001; //in ratio (1 pip=0.0001)
//double bid;
//double ask;
double inter_profit;
double profit;
double GROI;
double ROI;
double spread;
int position = 0; // type of position opened: 1 long/-1 short
int ticks_counter = 0;

string      stringThisDate;
datetime    thisTime;
MqlDateTime timeStruct;
MqlDateTime currentTimeStruct;
//string      filename;
int         filehandleTest;
int         filehandle_record;

int first_pos = 0;
int ticks_counter_open = 0;
int dif_ticks = 0;

//int firstBuff[2] = {1,1};

// Strategy variables


void openPosition(string origin, int thisPos){
   
   position = thisPos;
   Bi = bid;
   Ai = ask;
   B_sl = Bi;
   A_sl = Ai;
   string thisPosString;
   if(thisPos==1){
      m_Trade.Buy(lot,thisSymbol);
      stoploss = -Ai*sl_thr+Bi;
      thisPosString = "Long";
   }
   else{if(thisPos==-1){
      m_Trade.Sell(lot,thisSymbol);
      stoploss = Bi*sl_thr+Ai;
      thisPosString = "Short";
   }}
   
   deadline = 1; // reset deadline
   Bi_soll = buffer[bufferSize-1][0];
   Ai_soll = buffer[bufferSize-1][1];
   dif_ticks = ticks_counter-ticks_counter_open;
   Print(origin," ",thisPosString,". Number events ",nEventsPerStat," Ticks diff ",dif_ticks," Bi soll ",Bi_soll, " Bi ist ",Bi," Ai soll ",Ai_soll," Ai ",Ai," SL ",stoploss, " SP ",(Ai-Bi)/Ai);
}


void closePosition(string close_type){

   m_Trade.PositionClose(thisSymbol);// close position if deadline reached
   deadline = -1;
   
   if (position==1){
      GROI = 100*(bid-Bi)/Ai;
      ROI = 100*(bid-Ai)/Ai;

   }else{if(position==-1){
      GROI = 100*(Bi-bid)/ask;
      ROI = 100*(Bi-ask)/ask;
   }}
   
   profit = ROI*lot*lot_in_eur/100;
   spread = GROI-ROI;
   Print(close_type," ",position," Bi ",Bi," BiS ",Bi_soll," Ai ",Ai," AiS ",Ai_soll," Bo ",bid," Ao ",ask," spread ",spread," GROI ",GROI," ROI ",ROI," Profit ", profit," dif ticks ",dif_ticks);
   // send signal to trading manager that position has been closed
   if(close_type=="LC"){
      filename = "CL";
      
   }else{
      filename = close_type;
   }
   filehandleTest = FileOpen(directoryNameLive+filename,FILE_WRITE|FILE_CSV,',');
   FileWrite(filehandleTest,thisSymbol,toc,TimeCurrent(),position,Bi,Ai,bid,ask,dif_ticks,GROI,spread,ROI,profit);
   FileClose(filehandleTest);
   position = 0;

}

int OnInit()
  {
//---
   file_index = 0;
   thisSymbol = Symbol();
   directoryNameLive = "IOlive//"+thisSymbol+"//";
   directoryNameRecordings = "Data//"+thisSymbol+"//";
   //Print(thisSymbol+" fetcher launched");
   
   timer_bool = EventSetMillisecondTimer(10);

   TTfile = "TT";// trade trigger file signalling
   LCfile = "LC";
   // set milisecond timer
   //timer_bool = EventSetMillisecondTimer(10);
   Print(thisSymbol+" Fetcher-Trader-Recorder version 1.0 launched");
   
   // Test to check open/close instants and synchronizity with Python
   
   // variables
   thisTime = TimeCurrent(); // get Init recording time
   ResetLastError(); 
   // Generate time structure
   if(TimeToStruct(thisTime,timeStruct)==false){
      PrintFormat("TimeToStruct falied. Error code %d",GetLastError());
   }

   // Generate file name
   stringThisDate = StringFormat("%4d%02d%02d%02d%02d%02d",timeStruct.year,timeStruct.mon,timeStruct.day,timeStruct.hour,timeStruct.min,timeStruct.sec);
   filename = thisSymbol+"_"+stringThisDate+".txt";
   //Print(stringThisDate);
   // init file
   filehandle_record=FileOpen(directoryNameRecordings+"//"+filename,FILE_WRITE|FILE_CSV,',');
   FileWrite(filehandle_record,"DateTime","SymbolBid","SymbolAsk");
   
   
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   Print(thisSymbol+" fetcher closed");
   FileClose(filehandle);
   
   // check if position is opened
   if (m_Position.Select(thisSymbol)){
      m_Trade.PositionClose(thisSymbol);// close position
   }
   
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
  
   /*****************************
   *********** FETCHER **********
   ******************************/
   
   // new Event comes. Get bids, asks and datetimes
   
   currTime = TimeCurrent();
   bid = SymbolInfoDouble(Symbol(),SYMBOL_BID);
   ask = SymbolInfoDouble(Symbol(),SYMBOL_ASK);
   
   ticks_counter = ticks_counter+1;
   /*
   if(position==0 && first_pos==0){
      Print("Bid ",bid," ask ",ask," Ticks counter ",ticks_counter);
   }*/

   //ArrayPrint(bid_counters);
   
   // if bids counter is reset=>
   // init and open new buffer
   if(bids_counter==0){
      // Generate file name
      filename = thisSymbol+"_"+StringFormat("%1d",file_index)+".txt";
         
      //Print("Open "+filename);
      // init file
      filehandle = FileOpen(directoryNameLive+filename,FILE_WRITE|FILE_CSV,',');
      FileWrite(filehandle,"DateTime","SymbolBid","SymbolAsk");
      
      //if(position==0 && first_pos==0){
       //  Print("Ticks counter ",ticks_counter);
      //}
   }
   
   // save new entry in buffer
   FileWrite(filehandle,TimeCurrent(),bid,ask);
   
   buffer[bids_counter][0] = bid;
   buffer[bids_counter][1] = ask;
   
   // update bids counter
   bids_counter = (bids_counter+1)%bufferSize;
   
   // close buffer if bids counter reset
   if(bids_counter==0){
      //Print("Close "+filename);
      tic = TimeCurrent();
      ticks_counter_open = ticks_counter;
      FileClose(filehandle);  
      file_index = (file_index+1)%nBuffers;
   }
   
   /****************************
   ********** RECORDER *********
   *****************************/
   
   // permanent-record data
   if(record_data){
      
      if(TimeToStruct(currTime,currentTimeStruct)==false){
         PrintFormat("TimeToStruct falied. Error code %d",GetLastError());
      }
      
      // change of days
      if(currentTimeStruct.day!=timeStruct.day){
         
         // close previous file
         FileClose(filehandle_record); 
         // create new file
         //thisTime = TimeCurrent();
         //ResetLastError(); 
         // Generate time structure
         if(TimeToStruct(currTime,timeStruct)==false){
            PrintFormat("TimeToStruct falied. Error code %d",GetLastError());
         }
         // Generate file name
         stringThisDate = StringFormat("%4d%02d%02d%02d%02d%02d",timeStruct.year,timeStruct.mon,timeStruct.day,timeStruct.hour,timeStruct.min,timeStruct.sec);
         filename = thisSymbol+"_"+stringThisDate+".txt";
      
         filehandle_record=FileOpen(directoryNameRecordings+"//"+filename,FILE_WRITE|FILE_CSV,',');
         FileWrite(filehandle_record,"DateTime","SymbolBid","SymbolAsk");
      }
      
      //FileWrite(filehandle,TimeCurrent(),bid,ask);
      //Print(filehandle);
      
      FileWrite(filehandle_record,currTime,bid,ask);
      
   }
   
   /****************************
   *********** TRADER **********
   *****************************/
   
   
   // Launch position 
   
   //for(int nn=0; nn<nNets; nn++){
      //+StringFormat("%1d",nn)
      if(FileIsExist(directoryNameLive+TTfile)==1 ){
         // open position
            first_pos = 1;
            
            
            // Read type of trigger and delete TT
            int k;
            bool reset = false;
            do{
               filehandle_trader = FileOpen(directoryNameLive+TTfile,FILE_READ);
               predictionString = FileReadString(filehandle_trader);
               k = StringSplit(predictionString,StringGetCharacter(",",0),chunks);
               FileClose(filehandle_trader);
            }while(k!=3);
            prediction = StringToInteger(chunks[0]);
            lot = StringToDouble(chunks[1]);
            // extend deadline only when new deadline is further in time
            if(nEventsPerStat-deadline<StringToDouble(chunks[2])){
               nEventsPerStat = StringToDouble(chunks[2]);
               reset = true;
            }
            
            FileDelete(directoryNameLive+TTfile);
            
            // Record datetime bid and ask for checking
            //bid = SymbolInfoDouble(Symbol(),SYMBOL_BID);
            //ask = SymbolInfoDouble(Symbol(),SYMBOL_ASK);
            //FileWrite(filehandleTest,TimeCurrent(),bid,ask);
            
            diffTime = toc-tic;
            // launch long positon
            if(prediction>0){
               //if the position for this symbol already exists -> extend deadline
               if(m_Position.Select(thisSymbol)){
                  
                  //if(m_Position.PositionType()==POSITION_TYPE_SELL) m_Trade.PositionClose(my_symbol);  //and this is a Sell position, then close it
                  if(m_Position.PositionType()==POSITION_TYPE_BUY){
                     if(reset){
                        Print("Deadline extended by ",nEventsPerStat);
                        dif_ticks = dif_ticks+ticks_counter-ticks_counter_open;
                        // reset deadline
                        deadline = 1;
                     }else{
                        Print("Deadline not extended. Remaining ",nEventsPerStat-deadline);
                     }
                     
                  }
               }else{ // open new long position
                  toc = TimeCurrent();
                  openPosition("TICK", 1);
               }
               
            }
            // launch short positon
            else{
               //if the position for this symbol already exists -> extend deadline
               if(m_Position.Select(thisSymbol)){
                  
                  if(m_Position.PositionType()==POSITION_TYPE_SELL){ 
                     if(reset){
                        dif_ticks = dif_ticks+ticks_counter-ticks_counter_open;
                        Print("Deadline extended by ",nEventsPerStat);
                        // reset deadline
                        deadline = 1;
                     }else{
                        Print("Deadline not extended. Remaining ",nEventsPerStat-deadline);
                     }
                  }
               }
               // open new position
               else{
                  toc = TimeCurrent();
                  openPosition("TICK", -1);
               }
               
               
            }
            
      }
      
      // check if flag close position has been launched
      if(FileIsExist(directoryNameLive+LCfile)==1 ){
         FileDelete(directoryNameLive+LCfile);
         closePosition("LC");
      }
 
      // Check state of open position
      if(deadline>0){

         // update deadline
         deadline = (deadline+1)%(nEventsPerStat-dif_ticks);
         
         // close position if deadline reached
         if (deadline==0){
            //deadline[nn] = -1;
            closePosition("CL");
            
         }
         
         else{
            // check if stoploss reached
            if((position==1 && bid<stoploss) || (position==-1 && ask>stoploss)){
               closePosition("SL"); 
            }
            /*
            // Update stoploss
            if (position==1 && bid>B_sl){
               B_sl = bid;
               A_sl = ask;
               stoploss = -A_sl*sl_thr+B_sl;
               Print("SL=",stoploss);
            }
            else{if(position==-1 && ask<A_sl){
               B_sl = bid;
               A_sl = ask;
               stoploss = B_sl*sl_thr+A_sl;
               Print("SL=",stoploss);
            }}
            */
         }
         
         
      }
     //}
  }
  
void OnTimer(){

   /****************************
   *********** TRADER **********
   *****************************/
   
   
   // Launch position 
   
   //for(int nn=0; nn<nNets; nn++){
   
      if(FileIsExist(directoryNameLive+TTfile)==1 ){
         // open position
            first_pos = 1;
            
            // Read type of trigger and delete TT
            
            int k;
            bool reset = false;
            do{
               filehandle_trader = FileOpen(directoryNameLive+TTfile,FILE_READ);
               predictionString = FileReadString(filehandle_trader);
               k = StringSplit(predictionString,StringGetCharacter(",",0),chunks);
               FileClose(filehandle_trader);
            }while(k!=3);
            prediction = StringToInteger(chunks[0]);
            lot = StringToDouble(chunks[1]);
            // extend deadline only when new deadline is further in time
            if(nEventsPerStat-deadline<StringToDouble(chunks[2])){
               nEventsPerStat = StringToDouble(chunks[2]);
               reset = true;
            }
            FileDelete(directoryNameLive+TTfile);
            
            // Record datetime bid and ask for checking
            //bid = SymbolInfoDouble(Symbol(),SYMBOL_BID);
            //ask = SymbolInfoDouble(Symbol(),SYMBOL_ASK);
            //FileWrite(filehandleTest,TimeCurrent(),bid,ask);
            //toc = TimeCurrent();
            diffTime = toc-tic;
            // launch long positon
            if(prediction>0){
               //if the position for this symbol already exists -> extend deadline
               if(m_Position.Select(thisSymbol)){
                  
                  //if(m_Position.PositionType()==POSITION_TYPE_SELL) m_Trade.PositionClose(my_symbol);  //and this is a Sell position, then close it
                  if(m_Position.PositionType()==POSITION_TYPE_BUY){ 
                     if(reset){
                        dif_ticks = dif_ticks+ticks_counter-ticks_counter_open;
                        Print("Deadline extended by ",nEventsPerStat);
                        // reset deadline
                        deadline = 1;
                     }else{
                        Print("Deadline not extended. Remaining ",nEventsPerStat-deadline);
                     }
                     
                  }
               }else{ // open new position
                  toc = TimeCurrent();
                  openPosition("TIMER", 1);
               }
               
            }
            // launch short positon
            else{
               //if the position for this symbol already exists -> extend deadline
               if(m_Position.Select(thisSymbol)){
                  
                  if(m_Position.PositionType()==POSITION_TYPE_SELL){ 
                     if(reset){
                        dif_ticks = dif_ticks+ticks_counter-ticks_counter_open;
                        Print("Deadline extended by ",nEventsPerStat);
                        // reset deadline
                        deadline = 1;
                     }else{
                        Print("Deadline not extended. Remaining ",nEventsPerStat-deadline);
                     }
                  }  
                  /*
                  else{
                     // WARNING! Expected bear market when position is long!!!
                     Print("Warning! Expected bear market when position is long. Closing and opening");
                     // close position
                     m_Trade.PositionClose(thisSymbol);// close position if deadline reached
                     profit = bid-openedPosBid-spread;
                     Print("Profit = ", profit);
                     // open short
                     position = -1;
                     m_Trade.Sell(Lot,thisSymbol);
                     openedPosBid = SymbolInfoDouble(Symbol(),SYMBOL_BID);
                     openedPosAsk = SymbolInfoDouble(Symbol(),SYMBOL_ASK);
                     stoplossBid = openedPosBid;
                     stoplossAsk = openedPosAsk;
                     stoploss = 1000*(openedPosAsk-openedPosBid);
                     deadline[nn] = 1; // reset deadline

                  }*/
                  //if(m_Position.PositionType()==POSITION_TYPE_BUY) return; 
               }
               // open new position
               else{
                  toc = TimeCurrent();
                  openPosition("TIMER", -1);
               }
               
               
            }
            
      }
      // check if flag close position has been launched
      if(FileIsExist(directoryNameLive+LCfile)==1 ){
         FileDelete(directoryNameLive+LCfile);
         closePosition("LC"); 
      }
     //}

}
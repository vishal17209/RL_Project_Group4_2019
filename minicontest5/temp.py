# Solicit an action
            action = None
            self.mute(agentIndex)
            if self.catchExceptions:
                try:
                    # timed_func = TimeoutFunction(agent.getAction, int(self.rules.getMoveTimeout(agentIndex)) - int(move_time))
                    timed_func = TimeoutFunction(agent.getAction, int(self.state.data.score))
                    try:
                        start_time = time.time()
                        if skip_action:
                            raise TimeoutFunctionException()
                        action = timed_func( observation )
                    except TimeoutFunctionException:
                        print("Agent %d timed out on a single move!" % agentIndex, file=sys.stderr)
                        self.agentTimeout = True
                        self._agentCrash(agentIndex, quiet=True)
                        self.unmute()
                        return

                    move_time += time.time() - start_time

                    # if move_time > self.rules.getMoveWarningTime(agentIndex):
                    #     self.totalAgentTimeWarnings[agentIndex] += 1
                    #     print("Agent %d took too long to make a move! This is warning %d" % (agentIndex, self.totalAgentTimeWarnings[agentIndex]), file=sys.stderr)
                    #     if self.totalAgentTimeWarnings[agentIndex] > self.rules.getMaxTimeWarnings(agentIndex):
                    #         print("Agent %d exceeded the maximum number of warnings: %d" % (agentIndex, self.totalAgentTimeWarnings[agentIndex]), file=sys.stderr)
                    #         self.agentTimeout = True
                    #         self._agentCrash(agentIndex, quiet=True)
                    #         self.unmute()
                    #         return

                    self.totalAgentTimes[agentIndex] += move_time
                    #print("Agent: %d, time: %f, total: %f" % (agentIndex, move_time, self.totalAgentTimes[agentIndex]))
                    # if self.totalAgentTimes[agentIndex] > self.rules.getMaxTotalTime(agentIndex):
                    #     print("Agent %d ran out of time! (time: %1.2f)" % (agentIndex, self.totalAgentTimes[agentIndex]), file=sys.stderr)
                    #     self.agentTimeout = True
                    #     self._agentCrash(agentIndex, quiet=True)
                    #     self.unmute()
                    #     return
                    self.unmute()
                except Exception as data:
                    self._agentCrash(agentIndex)
                    self.unmute()
                    return
            else:
                # try:
                # timed_func = TimeoutFunction(agent.getAction, int(math.ceil(self.state.data.score / (SCALING_FACTOR+1))))
                # try:
                #     start_time = time.time()
                #     action = timed_func(observation)
                # except TimeoutFunctionException:
                #     print('You have run out of compute time! You exceeded {:.3f}s of compute'.format(self.state.data.score / (SCALING_FACTOR+1)))
                #     self.state.data.score = 0
                #     self.state.data._lose = True
                #     self.rules.process(self.state, self)
                #     continue
                # # except:
                # #     print('Your agent crashed!')
                # #     self.state.data.score = 0
                # #     self.state.data._lose = True
                # #     self.rules.process(self.state, self)
                # #     continue
                # move_time = time.time() - start_time
                action = agent.getAction(self.state.deepCopy())
                assert(action in self.state.getLegalActions(agentIndex)), str(self.state) + " "+ str(self.state.getLegalActions(agentIndex)) +" "+str(action) +" " + str(agentIndex)
                #whoami

            self.unmute()

    self.state.data.score += 0 #whoami max(0,-1 * SCALING_FACTOR)
            # if self.state.data.score <= 0:
            #     self.state.data.score = 0
            #     self.state.data._lose = True
            #     self.rules.process(self.state, self)
            #     continue #whoami
            # if self.state.data.deathCount >= 2:
            #     self.state.data._lose = True
            #     self.rules.process(self.state, self)
            #     continue
            #     #whoami

            # Execute the action
            self.moveHistory.append( (agentIndex, action) )
            if self.catchExceptions:
                try:
                    self.state = self.state.generateSuccessor( agentIndex, action )
                except Exception as data:
                    self.mute(agentIndex)
                    self._agentCrash(agentIndex)
                    self.unmute()
                    return
            else:
                self.state = self.state.generateSuccessor( agentIndex, action )



Average Score: 514.4871641791193
Win Rate:      1935/2010 (0.96)

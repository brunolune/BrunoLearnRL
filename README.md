# LearnRL
DataScientest Bootcamp - RL project 

This is a copy of UC Berkely famous AI Pacman programs !!

I have completed the first assignment concerning valueIterationAgents.py that examplifies Dynamic programming for a simple example. 

I think it will be a good set of programs to start learning about Reinforcement Learning.

Bruno




L'ambition de ce projet etait de se familiariser avec les methodes du reinforcement learning (apprentissage renforce) en entrainant un "agent" a jouer a un jeu simple.
Apres avoir etudier la theorie du reinforcement learning (RL) avec le livre "intro to reinforcement learning" de Barto et Sutton, notre choix s'est arrete sur le Pacman du cours intro to AI de UC Berkeley (CS188). Les raisons: la structure de ces programmes et les differents agents illustrent bien le developpemnt des concepts et la crewation d'agent de plus en plus elabore comme decrit dans le livre de sutton et barto' D'autre l'interet de ces programmes est qu'ils sont ecrit en python objet oriente. La structure hierarchique des classes refletent bien l'interdependence des concpts dans la theorie. Voir ump graphe mdp<-valueestimationagent<-reinforcement agent<-...'
'


La familiarisation aux concepts du RL se fait en plusieurs etapes avec le cours deUC Berkeley
premiere etape: se familiariser avec les concepts de mdp et dynamic programming:
mdp (markov decision process) 

the theory presented in the reference book intro to reinforcement learning from Sutton and barto goes hand in hand with the developpement of the programs of intro to 


Objectif: entrainer un agent a evolue dde facon optimale dans un environnment 
comment ca marche RL?
RL consiste a attribuer a tous les etats de l'environnement une valeur V(s) suivant son potentiel de generer un bon score dans le futur'


A la base il y a MDP (Markov Decision process), le MDP conceptualise l'interaction d'un agent avec son environnement qui consiste en la succession de etat (S=State), action (A)et Recompense (R=reward), donnant lieu a la serie S0,A0,R1,S1,A1 ...' En outre, le MDP suppose l'existence de probabilites de transition entre les etats et des possibilites de recompense suivant les actions de l'agent.p(s',r|s,a). MDP comprend aussi la definition d'un facteur de devaluation limitant l''importance des recompenses dans un futurr lointain.' Point important, la valeur d'un etat determine par un MDP ne prend en compte que l'etat present et futur du systeme (the future is independent of the past given the present: c'est l'idee qu'on accumule l'experience dans la valeur de l'etat ie de la qvalues.)'

Comment determine t on la facon optimale d'evoluer dans l'environnement?
'

)'

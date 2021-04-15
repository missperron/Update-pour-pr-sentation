%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% energieSolaire.m
% Fonction calculant l'énergie solaire produite par les panneaux (kW)
% Inputs : - irrMoy (1x24) (vecteur de l'irraditiation moyenne par heure dans une
% journée)
% - tempMoy (1x24) (vecteur de la température moyenne par heure
% Output : - pvMoy (1x24) (vecteur de la puissance solaire produite par heure)
% Auteurs : Dominic Rivest
% Date de création : 2021-04-08
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function pvMoy = energieSolaire(irrMoy, tempMoy)

% Paramètres des panneaux solaires
N_pv = 1; % Nombre de panneaux (1 normal)
eta_pv = 0.23; % Effiacité des panneaux
theta_stc = 25; % Température de référence
P_stc = 1; % Puissance crête aux conditions standards (kW)
I_stc = 1000; % Irradiation solaire dans les conditions standards (W/m2)
C_T = 0.0045; % Coefficient de température
eps_pv = 0; % Coefficient de correction

% Initialisation des variables
pvMoy = []; % Puissance produite par heure (kW)
theta_cell = []; % Température du panneau (c)
for i = 1:length(irrMoy)
    theta_cell = tempMoy(i)+irrMoy(i)/I_stc*eps_pv;
    pvMoy(i) = N_pv*eta_pv*P_stc*irrMoy(i)/I_stc*(1+(theta_cell-theta_stc)*C_T);
end

end
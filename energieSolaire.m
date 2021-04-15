%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% energieSolaire.m
% Fonction calculant l'�nergie solaire produite par les panneaux (kW)
% Inputs : - irrMoy (1x24) (vecteur de l'irraditiation moyenne par heure dans une
% journ�e)
% - tempMoy (1x24) (vecteur de la temp�rature moyenne par heure
% Output : - pvMoy (1x24) (vecteur de la puissance solaire produite par heure)
% Auteurs : Dominic Rivest
% Date de cr�ation : 2021-04-08
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function pvMoy = energieSolaire(irrMoy, tempMoy)

% Param�tres des panneaux solaires
N_pv = 1; % Nombre de panneaux (1 normal)
eta_pv = 0.23; % Effiacit� des panneaux
theta_stc = 25; % Temp�rature de r�f�rence
P_stc = 1; % Puissance cr�te aux conditions standards (kW)
I_stc = 1000; % Irradiation solaire dans les conditions standards (W/m2)
C_T = 0.0045; % Coefficient de temp�rature
eps_pv = 0; % Coefficient de correction

% Initialisation des variables
pvMoy = []; % Puissance produite par heure (kW)
theta_cell = []; % Temp�rature du panneau (c)
for i = 1:length(irrMoy)
    theta_cell = tempMoy(i)+irrMoy(i)/I_stc*eps_pv;
    pvMoy(i) = N_pv*eta_pv*P_stc*irrMoy(i)/I_stc*(1+(theta_cell-theta_stc)*C_T);
end

end
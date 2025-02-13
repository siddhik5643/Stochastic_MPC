Ts = 0.1;
x0 = zeros(15,1);
u0 = [-0.65*9.81; 0; 0; 0];
[Ac,Bc] = symLin(x0,u0);
Ac = double(Ac);
Bc = double(Bc);
Cc = eye(15);
sys = ss(Ac,Bc,Cc,0);
sysd = c2d(sys,Ts);
[Ad, Bd, Cd, Dd] = ssdata(sysd);

 
%%
%linear model
import casadi.*
mpc = import_mpctools();

Nx = 15;  % number of x variable 
Nu = 4;  % number of control variable
Nt = 60; % prediction horizon N 
Delta = 0.1; % timestep
Nsim= 1000; % simulation time
N = struct('x', Nx, 'u', Nu, 't', Nt); % define the dimension of the MPC

x = NaN(Nx, Nsim + 1); % x is used to store the x variable
x(:,1)=[0; 0; 0; 0; 0.0; 0.0; 0; 0; 0; 0; 0; 0; 0; 0; 0]; % initial value of x
u = NaN(Nu, Nsim); % u is used to store the control variable

for t=0:Nsim
% define the dynamic system (equality constriants)
f = mpc.getCasadiFunc(@linear_ode, [Nx, Nu], {'x', 'u'},'Delta', Delta);
% define the stage cost of MPC
l = mpc.getCasadiFunc(@stagecost, [Nx, Nu], {'x', 'u'}, {'l'});
% define the terminal cost of MPC
Vf = mpc.getCasadiFunc(@termcost, [Nx], {'x'}, {'Vf'});
% set the hard constraints of x and u
lbx = [-1000*ones(1,Nt+1); -1000*ones(1, Nt+1); -1000*ones(1, Nt+1); -100*ones(1, Nt+1); -100*ones(1, Nt+1); -100*ones(1, Nt+1); -pi/2*ones(1, Nt+1); -pi/2*ones(1, Nt+1); -pi*ones(1, Nt+1) ; -1000*ones(1, Nt+1); -1000*ones(1, Nt+1); -1000*ones(1, Nt+1); -1000*ones(1, Nt+1); -1000*ones(1, Nt+1); -1000*ones(1,Nt+1)];
ubx = [1000*ones(1,Nt+1); 1000*ones(1, Nt+1); 1000*ones(1, Nt+1); 100*ones(1, Nt+1); 100*ones(1, Nt+1); 100*ones(1, Nt+1); pi/2*ones(1, Nt+1); pi/2*ones(1, Nt+1); pi*ones(1, Nt+1); 1000*ones(1,Nt+1); 1000*ones(1,Nt+1); 1000*ones(1,Nt+1); 1000*ones(1,Nt+1); 1000*ones(1,Nt+1); 1000*ones(1,Nt+1)];
lbu = [ 0*ones(1,Nt); -22.52*ones(1,Nt); -22.52*ones(1,Nt); -1.08*ones(1,Nt)];
ubu = [ 90*ones(1,Nt); 22.52*ones(1,Nt); 22.52*ones(1,Nt); 1.08*ones(1,Nt)];
% define the solver    
commonargs = struct('l', l, 'Vf', Vf, 'lb', struct('u', lbu,'x',lbx),'ub', struct('u', ubu, 'x', ubx));
solvers = mpc.nmpc('f', f, 'N', N,'Delta', Delta, '**', commonargs);
% set the initial value    
r = [0;0;0];
if(t>Nsim/2)
  r = [2;0;0]; 
end
x(13:15,t+1) = r;

solvers.fixvar('x', 1, x(:,t+1));
% solve the MPC at time t
solvers.solve();
% show the status of solving optimization
fprintf('%d: %s\n', t, solvers.status);
if ~isequal(solvers.status, 'Solve_Succeeded')
   warning('%s failed at time %d!', t);            
end
% save the results as the guess solution for next timestep (for warm-start)
solvers.saveguess();
% save the results of x and u
u(:,t+1) = solvers.var.u(:,1);
x(:,t+2)= solvers.var.x(:,2);    
end

%%
close all
set(0, 'DefaultLineLineWidth', 1.5);
% plot the figure
Time=0:Delta:((Nsim+1))*Delta;

figure(1)

subplot(3,1,1)
plot(Time,x(1,:))
hold on
trac_1 = [zeros((Nsim+2)/2,1); 2*ones((Nsim+2)/2,1)]';
plot(Time,trac_1,'-.')
hold on
xlabel('Time')
ylabel('x(1) = ϕ')
xlim([Time(1),10])

legend('MPC','setpoint','Fontsize',12)

subplot(3,1,2)
plot(Time,x(2,:))
hold 
trac_1 = [zeros((Nsim+2)/2,1); 0*ones((Nsim+2)/2,1)]';
plot(Time,trac_1,'-.')
hold on
xlabel('Time')
ylabel('x(2) = θ')
xlim([Time(1),10])


subplot(3,1,3)
plot(Time,x(3,:))
hold on
plot(Time,trac_1,'-.')
hold on
xlabel('Time')
ylabel('x(3) = ψ')
xlim([Time(1),10])

figure(2)

subplot(3,1,1)
plot(Time,x(13,:))
hold on
trac_1 = [zeros((Nsim+2)/2,1); 0.5*ones((Nsim+2)/2,1)]';
plot(Time,trac_1,'-.')
hold on
xlabel('Time')
ylabel('x(1) = ϕ')
xlim([Time(1),100])

legend('MPC','setpoint','Fontsize',12)

subplot(3,1,2)
plot(Time,x(14,:))
hold on
plot(Time,zeros(Nsim+2,1),'-.')
hold on
xlabel('Time')
ylabel('x(14) = θ')
xlim([Time(1),100])


subplot(3,1,3)
plot(Time,x(15,:))
hold on
trac_1 = [zeros((Nsim+2)/2,1); -0.5*ones((Nsim+2)/2,1)]';
plot(Time,trac_1,'-.')
hold on
xlabel('Time')
ylabel('x(15)')
xlim([Time(1),100])




%%

function dxdt = linear_ode(x,u)
    Ad = [    1.0000         0         0    0.0992         0         0         0    0.0488         0         0    0.0012         0         0         0         0;
         0    1.0000         0         0    0.0992         0    0.0488         0         0    0.0012         0         0         0         0         0;
         0         0    1.0000         0         0    0.0992         0         0         0         0         0         0         0         0         0;
         0         0         0    0.9847         0         0         0    0.9735         0         0    0.0328         0         0         0         0;
         0         0         0         0    0.9847         0    0.9735         0         0    0.0328         0         0         0         0         0;
         0         0         0         0         0    0.9847         0         0         0         0         0         0         0         0         0;
         0         0         0         0         0         0    1.0000         0         0    0.0552         0         0         0         0         0;
         0         0         0         0         0         0         0    1.0000         0         0    0.0552         0         0         0         0;
         0         0         0         0         0         0         0         0    1.0000         0         0    0.1506         0         0         0;
         0         0         0         0         0         0         0         0         0    0.2636         0         0         0         0         0;
         0         0         0         0         0         0         0         0         0         0    0.2636         0         0         0         0;
         0         0         0         0         0         0         0         0         0         0         0    2.1581         0         0         0;
         0         0         0         0         0         0         0         0         0         0         0         0    1.0000         0         0;
         0         0         0         0         0         0         0         0         0         0         0         0         0    1.0000         0;
         0         0         0         0         0         0         0         0         0         0         0         0         0         0    1.0000];
    Bd = [         0         0   -0.0010         0;
         0    0.0010         0         0;
   -0.0077         0         0         0;
         0         0   -0.0369         0;
         0    0.0369         0         0;
   -0.1527         0         0         0;
         0    0.1030         0         0;
         0         0   -0.1030         0;
         0         0         0   -0.5055;
         0    1.6937         0         0;
         0         0   -1.6937         0;
         0         0         0  -11.5811;
         0         0         0         0;
         0         0         0         0;
         0         0         0         0];
    dxdt=Ad*x + Bd*u;
end


% function dxdt = ode(x, u)
% 
%     J1 = 120;
%     J2 = 100;
%     J3 = 80;
% 
%     phi = x(1);
%     theta = x(2);
%     psi = x(3);
%     omega1 = x(4);
%     omega2 = x(5);
%     omega3 = x(6);
% 
%     array1 = [1, sin(phi)*tan(theta), cos(phi)*tan(theta);
%                        0, cos(phi),            -sin(phi);
%                        0, sin(phi)/cos(theta), cos(phi)/cos(theta)]*[omega1; omega2; omega3];
% 
% 
% 
%     omega_dot = [((J2 - J3)* omega2 * omega3/J1)  + (u(1)/J1);
%                  ((J3 - J1)* omega3 * omega1/J2)  + (u(2)/J2);
%                  ((J1 - J2)* omega1 * omega2/J3)  + (u(3)/J3)];
% 
%     rdot =[0;0;0];
%     dxdt = [array1; omega_dot; rdot];
% 
% end
function dstate = Quad_model(x,u)
Ix = 7.5e-3;
Iy = 7.5e-3;
Iz = 1.3e-2;
l = 0.23;
Ir = 6e-5;
kf = 3.13e-5;
Km = 7.5e-7;
m = 0.65;
g = 9.81;
ktx = 0.1;
kty = 0.1;
ktz = 0.1;
krx = 0.1;
kry = 0.1;
Krz = 0.1;
wr = 0;
u1 = u(1);
u2 = u(2);
u3 = u(3);
u4 = u(4);
dstate(1) =  x(4);
dstate(2) =  x(5);
dstate(3) =  x(6);
dstate(4) =  -(1/m)*(ktx*x(4) + u1*(sin(x(7))*sin(x(9)) + cos(x(7))*cos(x(9))*sin(x(8))));
dstate(5) =  -(1/m)*(kty*x(5) + u1*(sin(x(7))*cos(x(9)) + cos(x(7))*sin(x(9))*sin(x(8))));
dstate(6) =  -(1/m)*(ktz*x(6) -m*g + u1*(cos(x(7))*cos(x(8))));
dstate(7) =   x(10) + x(12)*cos(x(7))*tan(x(8)) + x(11)*sin(x(7))*x(8);
dstate(8) =   x(11)*cos(x(7)) - x(12)*sin(x(7));
dstate(9) =   x(12)*cos(x(7))/cos(x(8)) + x(11)*sin(x(7))/cos(x(8));
dstate(10) =  -(1/Ix)*(krx*x(10) - l*u2 + (Iz - Iy)*x(11)*x(12) + Ir*x(11)*wr);
dstate(11) =  -(1/Iy)*(kry*x(11) + l*u3 + (Ix - Iz)*x(10)*x(12) + Ir*x(10)*wr);
dstate(12) =  -(1/Iz)*(u4 - Krz*x(12) + (Ix - Iy)*x(10)*x(11));
dstate(13:15) = [0;0;0];
end

function l = stagecost(x, u)
  r = x(13:15);
  Q = diag([1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]);
  R = diag([1,1,1,1])*0.01;
  e = [x(1:3)-r(:);x(4:12)];
  l = e'*Q*e + u'*R*u;
end

function [A,B] = symLin(x0,u0)
   syms x [15 1];
   syms u [4 1];
   A_matrix = jacobian(Quad_model(x,u),x);
   A = subs(A_matrix, [x; u], [x0; u0]);
   B_matrix = jacobian(Quad_model(x,u),u);
   B = subs(B_matrix, [x; u], [x0; u0]);
   endm1DC`

function Vf = termcost(x)
  r = x(13:15);
  Q = diag([1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]);
  R = diag([1,1,1,1])*0.01;
  Ts = 0.1;
  x0 = zeros(15,1);
  u0 = [-0.65*9.81; 0; 0; 0];
  [Ac,Bc] = symLin(x0,u0);
  Ac = double(Ac(1:12,1:12));
  Bc = double(Bc(1:12,:));
  Cc = eye(12);
  sys = ss(Ac,Bc,Cc,0);
  sysd = c2d(sys,Ts);
  [Ad, Bd, Cd, Dd] = ssdata(sysd);
  Kinf = dlqr(Ad,Bd,Q,R);
  Pinf = dlyap((Ad + Bd*Kinf),(Q + Kinf'*R*Kinf));
  %[Pinf,u,t] = dare(Ad,Bd,Q,R)
  e = [x(1:3)-r(:);x(4:12)];
  Vf = e'*Pinf*e;
end
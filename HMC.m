function [result_q]=HMC(numFilter,lambdaF, filters,currSample, stepsize, L)

q=currSample;
sigma = 0.1;
p= normrnd(0,sigma,size(currSample));
current_p=p;

p=p- (stepsize/2) * grad_U(q, numFilter, filters, lambdaF);
for i=1:L
  q=q+ stepsize * p/sigma;
  if(i~=L) 
      p=p-stepsize * grad_U(q, numFilter, filters, lambdaF); 
  end
end

p=p-stepsize/2 *grad_U(q, numFilter, filters, lambdaF);


current_U=U(currSample, numFilter, filters, lambdaF);
current_K= sum(sum((current_p.^2/sigma)))/2;
  
proposed_U=U(q, numFilter, filters, lambdaF);
proposed_K=sum(sum((p.^2/sigma)))/2;

test=rand(1);
accProb = exp(current_U-proposed_U+current_K-proposed_K);
fprintf('accept probabilty: %f ', accProb);
if (test<accProb)
   result_q=q;
   disp( [ 'Accepted!' ]);
else
   result_q=currSample;
   disp( [ 'not accepted!' ]);
end
    
end

function g_U = grad_U(currImage, numFilter, filters, lambdaF)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% grad_U(currSample)
rSample = cell(numFilter,1);
for iFilter = 1:numFilter
    rSample{iFilter}=filter2(filters{iFilter},currImage);
end

g_U=zeros(size(currImage));
for iFilter = 1:numFilter
  delta_U= conv2(sign(rSample{iFilter}).*lambdaF{iFilter},filters{iFilter},'same'); 
  g_U =g_U + delta_U;
end
  g_U=-g_U;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end 

function result_u = U(currImage, numFilter, filters, lambdaF)
   rSample = cell(numFilter,1);
   for iFilter = 1:numFilter
    rSample{iFilter}=filter2(filters{iFilter},currImage);
   end

   result_u=0;
   for iFilter = 1:numFilter
      result_u = result_u + sum( sum(abs(rSample{iFilter}).*lambdaF{iFilter}));
   end
   
   result_u=-result_u;
   
end






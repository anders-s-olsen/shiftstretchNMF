# optimization loop
from tqdm import tqdm
import torch
import numpy as np
import copy

def Optimizationloop(data, model, optimizer=None, lr=None, scheduler=None, max_iter=10000, tol=1e-10,disable_output=False, num_comparison=50):

    if optimizer is None:
        if lr is None:
            lr = 0.1
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).train()

    # num_comparison = 50
    best_loss = np.inf
    best_model_list = []

    all_loss = []
    lrs = []
    pbar = tqdm(total=max_iter,disable=disable_output)

    for epoch in range(max_iter):
        loss = model(data)
        all_loss.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])

        if epoch%(num_comparison//5)==0:
            best_model_list.append(copy.deepcopy(model.get_model_params()))

        if epoch>=num_comparison:
            # for every 10 iterations, save the model parameters
            if epoch%(num_comparison//5)==0:
                best_model_list.pop(0)
            if scheduler is not None:
                scheduler.step(loss)
                if optimizer.param_groups[0]["lr"]<0.001:
                    break
            else: #specify relative tolerance threshold
                latest = np.array(all_loss[-num_comparison:])
                minval = np.min(latest)
                secondlowest = np.min(latest[latest!=minval])
                crit = (secondlowest-minval)/minval
                pbar.set_description('Loss: %.4f, Relative change: %.2e'%(all_loss[-1],crit))
                pbar.update(1)
                if crit<tol or all_loss[-1]==np.max(latest):
                    # set the model to the best model
                    best_model_list.append(copy.deepcopy(model.get_model_params()))
                    for best_model in best_model_list:
                        model.set_model_params(best_model)
                        loss = model(data)
                        if loss.item()<best_loss:
                            best_loss = loss.item()
                            best_model_params = best_model
                    model.set_model_params(best_model_params)
                    break
        else:
            pbar.set_description('Loss: %.4f: '%(all_loss[-1]))
            pbar.update(1)
    pbar.close()
    if epoch == max_iter-1:
        print("Max number of iterations reached OBS! Increase max_iter")
    else:
        print("Tolerance reached at " + str(epoch+1) + " number of iterations")
    # best_loss = min(all_loss)
    return all_loss, best_loss

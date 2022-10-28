import numpy as np
import matplotlib.pyplot as plt
import itertools
import copy
import sys

plt.rcParams['text.usetex'] = True

def split_norm(f, dt):
    g = np.array([e if e >= 0 else 0.0 for e in f])
    h = np.array([abs(e) if e <= 0 else 0.0 \
        for e in f])
    g_int = 0.5 * (2 * sum(g) - g[0] - g[-1]) * dt
    h_int = 0.5 * (2 * sum(g) - g[0] - g[-1]) * dt
    if( g_int > 0 ):
        g = g / g_int
    if( h_int > 0 ):
        h = h / h_int
    return g,h
    
def cdf(f, t):
    F = np.zeros(len(f))
    dt = t[1] - t[0]
#    total_integral = dt * sum(f)
    for i in range(len(f)):
        F[i] = F[i-1] + dt * f[i]
    return F

def quantile(F, t, p):
    G = np.zeros(len(p))
    idx = 1
    N = len(F)
    dt = t[1] - t[0]
    for (i,e) in enumerate(p):
        left = F[idx-1]
        right = F[idx]
        if( left >= e ):
            G[i] = t[idx]
            continue
        while( right < e and idx < N - 1 ):
            idx += 1
            left = right
            right = F[idx]

        if( idx == N - 1 ):
            for j in range(i,len(G)):
                G[j] = t[-1]
            break
        else:
            dF = right-left
            dt = t[idx] - t[idx-1]
            if( dF == 0 ):
                G[i] = left
            else:
                G[i] = t[idx-1] + dt/dF * (e-left)
            #G[i] = (1-alpha) * right + alpha * left
            
    return G
      
def w2(f,g,t,p):
    N = len(f)
    assert( N == len(g) )
    F = cdf(f,t)
    G = cdf(g,t)
    Finv = np.array(quantile(F, t, p))
    Ginv = np.array(quantile(G, t, p))
    diff = (Finv - Ginv)**2

    dp = p[1] - p[0]
    total_sum = 0.5 * dp * ( 2.0 * sum(diff) - diff[0] - diff[-1] )
    return total_sum

def ricker1(A, t0, sig):
    def helper(t):
        return A * (1-((t-t0)/sig)**2) * np.exp(-(t-t0)**2 / (2*sig**2))
    return helper

def ricker2(A, t0, sig):
    assert(len(A) == len(t0) and len(A) == len(sig))
    f = [ricker1(A[i], t0[i], sig[i]) for i in range(len(A))]
    def helper(t):
        val = 0.0
        for ff in f:
            val += ff(t)
        return val
    return helper

def shift_test_helper(**kw):
    A = kw['A']
    t0 = kw['t0']
    sig = kw['sig']
    shifts = kw['shifts']
    t = kw['t']
    p = kw['p']
    misfit = kw['misfit']
    do_plots = kw['do_plots']

    if( misfit == 'L2' ):
        dist = lambda f,g : sum((f-g)**2)
    else:
        dist = lambda f,g : w2(f,g,t,p)

    ref_ricker = ricker2(A,t0,sig)
    ref_array = np.array([ref_ricker(tt) \
        for tt in t])
    ref_p, ref_n = split_norm(ref_array, t[1]-t[0])
    plt.plot(t, ref_array, t, ref_p, t, ref_n)
    plt.savefig('ref.png')
    plt.clf()

    distances = []
    for (idx,shift) in enumerate(shifts):
        print('%d of %d'%(idx, len(shifts)))
        tshift = [t0[i] - shift[i] \
            for i in range(len(shift))]
        curr = ricker2(A,tshift,sig)
        curr_a = np.array([curr(tt) for tt in t])
        if( misfit == 'L2' ):
            distances.append( dist(curr_a,
                ref_array) )
        else:
            curr_p, curr_n = split_norm(curr_a,
                t[1]-t[0])
            distances.append( dist(curr_p, ref_p) \
                + dist(curr_n, ref_n) )
            if( do_plots ):
                plt.plot(t, curr_a) 
                plt.plot(t, curr_p, '--')
                plt.plot(t, curr_n, '*')
                plt.title("Shift (%f,%f,%s=%f)"%(
                    shift[0], shift[1], 
                    misfit, distances[-1]))
                plt.savefig('ricker%d.png'%idx)
                plt.clf()
    return np.array(distances)

def shift_test(num_shifts):
    shiftP = np.linspace(-3, 3, num_shifts)
    shiftS = np.linspace(-3, 3, num_shifts)
    d = { \
        'A': [1.0, 1.0],
        't0' : [-5.0, 5.0],
        'sig': [0.3, 0.3],
        't' : np.linspace(-15.0,15.0,1000),
        'p' : np.linspace(0,1,10000),
        'shifts' : np.array([e \
        for e in itertools.product(shiftP, shiftS)]),
        'misfit': 'W2',
        'do_plots' : False
    }
    l2_dict = copy.copy(d)
    l2_dict['misfit'] = 'L2'
    
    P, S = np.meshgrid(shiftP, shiftS)
    #S = np.transpose(S)
    distances = shift_test_helper(**d) \
        .reshape(P.shape)
    l2_distances = shift_test_helper(**l2_dict) \
        .reshape(P.shape)
    return P,S,distances,l2_distances

def do_shift_test():
    try:
        P = np.load('P.npy')
        S = np.load('S.npy')
        dW = np.load('dW.npy')
        dL = np.load('dL.npy')
    except:
        P,S,dW,dL = shift_test(int(sys.argv[1]))
        np.save('P.npy', P)
        np.save('S.npy', S)
        np.save('dW.npy', dW)
        np.save('dL.npy', dL)

    minw = min(np.ndarray.flatten(dW))
    maxw = min(np.ndarray.flatten(dW))
   
    minl = min(np.ndarray.flatten(dL))
    maxl = max(np.ndarray.flatten(dL))

    ext = [-3,3,-3,3]
    cm = 'jet'
    show_bar = True
  
    plt.figure()
    bar = plt.imshow(dW, extent=ext, cmap=cm,
        origin='lower')
    if( show_bar ):
        plt.colorbar(bar)
    plt.title(r'$W_2$ time shift test')
    plt.xlabel('P wave time shift')
    plt.ylabel('S wave time shift')
    plt.savefig('W2.png')
    plt.clf()

    plt.figure()
    bar = plt.imshow(dL, extent=ext, cmap=cm,
        origin='lower')
    if( show_bar ):
        plt.colorbar(bar)
    plt.title(r'$L^2$ time shift test')
    plt.xlabel('P wave time shift')
    plt.ylabel('S wave time shift')
    plt.savefig('L2.png')
    plt.clf()

def amp_test_helper(**kw):
    A = kw['A']
    t0 = kw['t0']
    sig = kw['sig']
    dA = kw['dA']
    t = kw['t']
    p = kw['p']
    misfit = kw['misfit']
    do_plots = kw['do_plots']

    if( misfit == 'L2' ):
        dist = lambda f,g : sum((f-g)**2)
    else:
        dist = lambda f,g : w2(f,g,t,p)

    ref_ricker = ricker2(A,t0,sig)
    ref_array = np.array([ref_ricker(tt) \
        for tt in t])
    ref_p, ref_n = split_norm(ref_array, t[1]-t[0])
    if( do_plots ):
         plt.figure()
         plt.plot(t, ref_array, t, ref_p, t, ref_n)
         plt.savefig('ref.png')
         plt.clf()

    distances = []
    for (idx,amp) in enumerate(dA):
        print('%d of %d'%(idx, len(dA)))
        da = np.array([A[i] * amp[i] \
            for i in range(len(amp))])
        curr = ricker2(da,t0,sig)
        curr_a = np.array([curr(tt) for tt in t])
        if( misfit == 'L2' ):
            distances.append( dist(curr_a,
                ref_array) )
        else:
            curr_p, curr_n = split_norm(curr_a,
                t[1]-t[0])
            distances.append( dist(curr_p, ref_p) \
                + dist(curr_n, ref_n) )
            if( do_plots ):
                plt.plot(t, curr_a) 
                plt.plot(t, curr_p, '--')
                plt.plot(t, curr_n, '*')
                plt.title("dA (%f,%f,%s=%f)"%(
                    da[0], da[1], 
                    misfit, distances[-1]))
                plt.savefig('rickerAmp%d.png'%idx)
                plt.clf()
    return np.array(distances)

def amp_test(num_amps):
    ampP = np.linspace(-2.0, 2.0, num_amps)
    ampS = np.linspace(-2.0, 2.0, num_amps)
  
    tau = 1e-3
    ampP = np.array([e if abs(e) > tau else \
        np.sign(e) * tau for e in ampP])
    ampS = np.array([e if abs(e) > tau else \
        np.sign(e) * tau for e in ampS])

    d = { \
        'A': [1.0, 1.0],
        't0' : [-5.0, 5.0],
        'sig': [0.3, 0.3],
        't' : np.linspace(-15.0,15.0,1000),
        'p' : np.linspace(0,1,10000),
        'dA' : np.array([e \
        for e in itertools.product(ampP, ampS)]),
        'misfit': 'W2',
        'do_plots': False
    }
    l2_dict = copy.copy(d)
    l2_dict['misfit'] = 'L2'
    
    P, S = np.meshgrid(ampP, ampS)
    #S = np.transpose(S)
    distances = amp_test_helper(**d) \
        .reshape(P.shape)
    l2_distances = amp_test_helper(**l2_dict) \
        .reshape(P.shape)
    return P,S,distances,l2_distances

def do_amp_test():
    try:
        P = np.load('Pamp.npy')
        S = np.load('Samp.npy')
        dW = np.load('dWamp.npy')
        dL = np.load('dLamp.npy')
    except:
        P,S,dW,dL = amp_test(int(sys.argv[1]))
        np.save('Pamp.npy', P)
        np.save('Samp.npy', S)
        np.save('dWamp.npy', dW)
        np.save('dLamp.npy', dL)

    pflat = np.ndarray.flatten(P)
    aP = min(pflat)
    bP = max(pflat)

    sflat = np.ndarray.flatten(S)
    aS = min(sflat)
    bS = max(sflat)

    ext = [aP,bP,aS,bS]
    cm = 'jet'
    show_bar = True
  
    plt.figure()
    bar = plt.imshow(dW, extent=ext, cmap=cm,
        origin='lower')
    if( show_bar ):
        plt.colorbar(bar)

    plt.title(r'$W_2$ partial dilation test')
    plt.xlabel('P dilation')
    plt.ylabel('S dilation')
    plt.savefig('W2amp.png')

    plt.clf()

    plt.figure()
    bar = plt.imshow(dL, extent=ext, cmap=cm,
        origin='lower')
    if( show_bar ):
        plt.colorbar(bar)
    plt.title(r'$L^2$ partial dilation test')
    plt.xlabel('P dilation')
    plt.ylabel('S dilation')
    plt.savefig('L2amp.png')
    plt.clf()

def noise_test_helper(**kw):
    A = kw['A']
    t0 = kw['t0']
    sig = kw['sig']
    noise = kw['noise']
    t = kw['t']
    p = kw['p']
    misfit = kw['misfit']
    do_plots = kw['do_plots']
    f_noise = kw['f_noise']

    if( f_noise == 'uniform' ):
        f_noise = lambda eta : lambda s : \
            eta * np.random.rand(s)
    elif( f_noise == 'gauss' ):
        f_noise = lambda eta : lambda s : \
            eta * np.random.randn(s)
    
    if( misfit == 'L2' ):
        dist = lambda f,g : sum((f-g)**2)
    else:
        dist = lambda f,g : w2(f,g,t,p)

    ref_ricker = ricker2(A,t0,sig)
    ref_array = np.array([ref_ricker(tt) \
        for tt in t])
    plt.plot(t, ref_array)
    plt.title('Reference Wave')
    plt.xlabel("Time")
    plt.ylabel("Displacement")
    plt.savefig('ref.png')
    ref_p, ref_n = split_norm(ref_array,
        t[1]-t[0])

    distances = []
    for (idx,eta) in enumerate(noise):
        print('%d of %d'%(idx, len(noise)))
        f = f_noise(eta)
        curr = f(len(ref_array)) + ref_array 
        if( misfit == 'L2' ):
            distances.append( dist(curr,
                ref_array) )
        else:
            curr_p, curr_n = split_norm(curr,
                t[1]-t[0])
            distances.append( dist(curr_p, ref_p) \
                + dist(curr_n, ref_n) )
            if( do_plots ):
                plt.plot(t, curr_a) 
                plt.plot(t, curr_p, '--')
                plt.plot(t, curr_n, '*')
                plt.title("dA (%f,%f,%s=%f)"%(
                    da[0], da[1], 
                    misfit, distances[-1]))
                plt.savefig('rickerAmp%d.png'%idx)
                plt.clf()
    return np.array(distances)

def noise_test(num_amps):
    noise = np.linspace(0.0,2.0,100)
    tmax = 15.0
    t = np.linspace(-tmax,tmax,100000)

    sin_noise = lambda eta : lambda s : \
        np.sin(2 * np.pi * eta / tmax * t)

    d = { \
        'A': [1.0, 1.0],
        't0' : [-5.0, 5.0],
        'sig': [0.3, 0.3],
        't' : t,
        'p' : np.linspace(0,1,10000),
        'noise' : noise,
        'misfit': 'W2',
        'do_plots': False,
        'f_noise' : 'uniform'
    }
    l2_dict = copy.copy(d)
    l2_dict['misfit'] = 'L2'
    
    distances = noise_test_helper(**d) 
    l2_distances = noise_test_helper(**l2_dict) 
    return noise,distances,l2_distances

def do_noise_test():
    try:
        noise = np.load('noise.npy')
        dW = np.load('dWnoise.npy')
        dL = np.load('dLnoise.npy')
    except:
        noise,dW,dL = noise_test(int(sys.argv[1]))
        np.save('noise.npy', noise)
        np.save('dWnoise.npy', dW)
        np.save('dLnoise.npy', dL)

    ext = []
    cm = 'jet'
    show_bar = True
  
    plt.figure()
    plt.plot(noise, dW)
    plt.title(r'$W_2$ noise test')
    plt.xlabel(r'SNR${}^{-1}$')
    plt.ylabel('Misfit')
    plt.savefig('W2noise.png')
    plt.clf()

    plt.figure()
    plt.plot(noise, dL)
    plt.title(r'$L^2$ noise test')
    plt.xlabel(r'SNR${}^{-1}$')
    plt.ylabel('Misfit')
    plt.savefig('L2noise.png')
    
if( __name__ == "__main__" ):
    do_shift_test()
    do_amp_test()
    do_noise_test()

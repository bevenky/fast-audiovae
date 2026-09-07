"""CPU-only synthetic parity and safety checks for experimental Apple block ops."""
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import numpy as np
from onnx import helper
from test_native_apple import AppleOperators, DOMAIN, model


class AppleBlockOperators(AppleOperators):
    # Inherited tests also protect the existing operator contract.
    def chain_model(self, seed=42, fused=True, dilation=9):
        rng=np.random.default_rng(seed);c=3;shape=['B',c,'T']
        arrays={'w':rng.normal(0,.15,(c,1,7)).astype(np.float32),'b':rng.normal(0,.1,c).astype(np.float32)}
        arrays.update({k:rng.uniform(.2,1.4,c).astype(np.float32) for k in ('ap','rp','aq','rq')})
        attr=dict(channels=c,native_abi=1,row_batches=0,backend=0,require_vforce=1)
        if fused:nodes=[helper.make_node('SnakeDW7SnakeF32',['x','w','b','ap','rp','aq','rq'],['y'],domain=DOMAIN,dilation=dilation,**attr)]
        else:nodes=[helper.make_node('SnakeF32',['x','ap','rp'],['pre'],domain=DOMAIN,**attr),
                    helper.make_node('CausalDW7SnakeF32',['pre','w','b','aq','rq'],['y'],domain=DOMAIN,dilation=dilation,**attr)]
        return model(nodes,arrays,{'x':shape},shape)

    def test_chain_boundaries_changing_lengths_causality_and_ownership(self):
        rng=np.random.default_rng(2103)
        for threads in (1,4):
            for dilation in (1,3,9):
                sessions=[self.session(self.chain_model(seed,True,dilation),threads) for seed in (42,43)]
                refs=[self.session(self.chain_model(seed,False,dilation),threads) for seed in (42,43)]
                inputs={t:rng.normal(0,.5,(2,3,t)).astype(np.float32) for t in (0,1,3,7,53,54,55,255,256,257,511,513)}
                for t in (513,0,1,3,7,53,54,55,255,256,257,511,513):
                    x=inputs[t];before=x.copy()
                    for i in (0,1,0):
                        with self.subTest(threads=threads,dilation=dilation,time=t,session=i):
                            y=sessions[i].run(None,{'x':x})[0];ref=refs[i].run(None,{'x':x})[0]
                            np.testing.assert_array_equal(y.view(np.uint32),ref.view(np.uint32))
                            np.testing.assert_array_equal(x.view(np.uint32),before.view(np.uint32))
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results=list(pool.map(lambda i:sessions[i].run(None,{'x':inputs[513]})[0],(0,1)))
                for i,y in enumerate(results):np.testing.assert_array_equal(y,refs[i].run(None,{'x':inputs[513]})[0])
                changed=inputs[513].copy();changed[...,257:]+=10
                np.testing.assert_array_equal(sessions[0].run(None,{'x':changed})[0][...,:257],results[0][...,:257])

    def test_bias_residual_exact_order_signed_zero_and_shape_rejection(self):
        shape=['B',3,'T'];bias=np.array([1e-8,-0.,-.75],np.float32)
        attrs=dict(channels=3,native_abi=1,row_batches=0,backend=0)
        graph=model([helper.make_node('BiasResidualF32',['product','bias','skip'],['y'],domain=DOMAIN,**attrs)],
                    {'bias':bias},{'product':shape,'skip':shape},shape)
        rng=np.random.default_rng(2104)
        for threads in (1,4):
            session=self.session(graph,threads)
            for t in (0,1,3,7,256,257,513):
                p=rng.normal(size=(2,3,t)).astype(np.float32);skip=rng.normal(size=p.shape).astype(np.float32)
                if t:p[:,1,0]=-0.;skip[:,1,0]=-0.
                expected=np.add(skip,np.add(p,bias[None,:,None],dtype=np.float32),dtype=np.float32)
                got=session.run(None,{'product':p,'skip':skip})[0]
                np.testing.assert_array_equal(got.view(np.uint32),expected.view(np.uint32))
            with self.assertRaisesRegex(Exception,'skip shape'):
                session.run(None,{'product':np.zeros((1,3,4),np.float32),'skip':np.zeros((2,3,4),np.float32)})

    def test_new_c_abi_safety_and_scalar_parity(self):
        ptr=ct.POINTER(ct.c_float);i64=ct.c_int64;i32=ct.c_int32
        triple=self.library.ncc_snake_dw7_snake_f32;triple.argtypes=[ptr]*8+[i64]*3+[i32]*3;triple.restype=i32
        add=self.library.ncc_bias_residual_f32;add.argtypes=[ptr]*4+[i64]*3+[i32]*2;add.restype=i32
        x=np.linspace(-1,1,513,dtype=np.float32);w=np.linspace(-.2,.2,7,dtype=np.float32)
        coeff=np.array([.7],np.float32);y=np.zeros_like(x);pp=lambda a:a.ctypes.data_as(ptr)
        args=[pp(x),pp(w),pp(coeff),pp(coeff),pp(coeff),pp(coeff),pp(coeff),pp(y)]
        self.assertEqual(triple(*args,1,1,513,9,0,1),0)
        for idx in range(7):
            bad=list(args);bad[7]=bad[idx];self.assertEqual(triple(*bad,1,1,513,9,0,1),3)
        self.assertEqual(triple(*([None]*8),0,1,513,9,0,1),0)
        self.assertEqual(triple(*args,1,1,513,2,0,1),1)
        self.assertEqual(triple(*args,-1,1,513,9,0,1),1)
        self.assertEqual(triple(*args,1,1,513,9,0,2),5)
        self.assertEqual(triple(*args,2**62,3,513,9,0,1),4)
        self.assertEqual(add(pp(x),pp(coeff),pp(x),pp(y),1,1,513,0,1),0)
        self.assertEqual(add(pp(x),pp(coeff),pp(x),pp(x),1,1,513,0,1),3)
        self.assertEqual(add(None,None,None,None,0,1,1,0,1),0)
        self.assertEqual(add(pp(x),None,pp(x),pp(y),1,1,513,0,1),1)
        pre=np.empty_like(x);ref=np.empty_like(x)
        snake=self.library.ncc_snake_f32;snake.argtypes=[ptr]*4+[i64]*3+[i32]*2;snake.restype=i32
        dw=self.library.ncc_dw7_snake_f32;dw.argtypes=[ptr]*6+[i64]*3+[i32]*3;dw.restype=i32
        for backend in (0,1,2):
            self.assertEqual(snake(pp(x),pp(coeff),pp(coeff),pp(pre),1,1,513,backend,1),0)
            self.assertEqual(dw(pp(pre),pp(w),pp(coeff),pp(coeff),pp(coeff),pp(ref),1,1,513,9,backend,1),0)
            self.assertEqual(triple(*args,1,1,513,9,backend,1),0)
            np.testing.assert_array_equal(y.view(np.uint32),ref.view(np.uint32))

    def test_new_op_bad_attributes_and_overridable_coefficients_rejected(self):
        graph=self.chain_model();next(a for a in graph.graph.node[0].attribute if a.name=='dilation').i=2
        with self.assertRaisesRegex(Exception,'dilation'):self.session(graph,1)
        graph=self.chain_model();graph.graph.input.append(helper.make_tensor_value_info('ap',1,[3]))
        with self.assertRaisesRegex(Exception,'non-overridable constant'):self.session(graph,1)

import { NDArray } from '../NDArray';
import { TypedArrayConstructor } from '../dtypes';
import { Where } from './indexes';
export type AxisArg = number | null;
export declare function kwargs_decorator<Wrapper extends (...args: any[]) => NDArray_, Implementation extends (...args: any[]) => NDArray>({ defaults, implementation, parsers, this_as_first_arg }: {
    defaults: [string, any][];
    implementation: Implementation;
    parsers?: ((kwargs: any) => void)[];
    this_as_first_arg?: boolean;
}): Wrapper;
type NDArray_non_0D = NDArray<any> | number[];
export declare namespace Func_a_q_axis {
    type Implementation = (a: NDArray<any>, q: number, axis: number) => NDArray<any>;
    type Kwargs = {
        a?: NDArray_non_0D;
        q?: number;
        axis?: AxisArg;
    };
    type Wrapper = (a: NDArray_non_0D | Kwargs, q: number | Kwargs, axis?: AxisArg | Kwargs) => NDArray<any>;
    const decorator: ({ defaults, implementation, parsers, this_as_first_arg }: {
        defaults: [string, any][];
        implementation: Implementation;
        parsers?: ((kwargs: any) => void)[];
        this_as_first_arg?: boolean;
    }) => Wrapper;
    const defaults: [string, any][];
    const parsers: ((kwargs: any) => void)[];
    const defaultDecorator: (implementation: Implementation) => Wrapper;
}
export declare namespace Func_a_lastAxis {
    type Implementation = (a: NDArray_, axis: number) => NDArray<any>;
    type Kwargs = {
        a?: NDArray_;
        axis?: AxisArg;
    };
    type Wrapper = (a: NDArray_ | Kwargs, axis?: AxisArg | Kwargs) => NDArray<any>;
    const decorator: ({ defaults, implementation, parsers, this_as_first_arg }: {
        defaults: [string, any][];
        implementation: Implementation;
        parsers?: ((kwargs: any) => void)[];
        this_as_first_arg?: boolean;
    }) => Wrapper;
    const defaults: [string, any][];
    const parsers: ((kwargs: any) => void)[];
    const defaultDecorator: (implementation: Implementation) => Wrapper;
}
export declare namespace Func_a_out {
    type Implementation = (a: NDArray<any>, out: NDArray<any> | null) => NDArray<any>;
    type Kwargs = {
        a?: NDArray_;
        out?: NDArray<any> | null;
    };
    type Wrapper = (a: NDArray_ | Kwargs, out?: NDArray<any> | null | Kwargs) => NDArray<any>;
    const decorator: ({ defaults, implementation, parsers, this_as_first_arg }: {
        defaults: [string, any][];
        implementation: Implementation;
        parsers?: ((kwargs: any) => void)[];
        this_as_first_arg?: boolean;
    }) => Wrapper;
    const defaults: [string, any][];
    const parsers: ((kwargs: any) => void)[];
    const defaultDecorator: (implementation: Implementation) => Wrapper;
}
export declare namespace Method_out {
    type Implementation = Func_a_out.Implementation;
    type Kwargs = {
        out?: NDArray<any> | null;
    };
    type Wrapper<T extends TypedArrayConstructor = Float64ArrayConstructor> = (out?: NDArray<any> | null | Kwargs) => NDArray<T>;
    const defaults: [string, any][];
    const parsers: ((kwargs: any) => void)[];
    function defaultDecorator<T extends TypedArrayConstructor = Float64ArrayConstructor>(implementation: Implementation): Wrapper<T>;
}
export declare namespace Func_a_decimals_out {
    type Implementation = (a: NDArray<any>, decimals: number, out: NDArray<any> | null) => NDArray<any>;
    type Kwargs = {
        a?: NDArray_;
        decimals?: number;
        out?: NDArray<any> | null;
    };
    type Wrapper = (a: NDArray_ | Kwargs, decimals: number | Kwargs, out?: NDArray<any> | null | Kwargs) => NDArray<any>;
    const decorator: ({ defaults, implementation, parsers, this_as_first_arg }: {
        defaults: [string, any][];
        implementation: Implementation;
        parsers?: ((kwargs: any) => void)[];
        this_as_first_arg?: boolean;
    }) => Wrapper;
    const defaults: [string, any][];
    const parsers: ((kwargs: any) => void)[];
    const defaultDecorator: (implementation: Implementation) => Wrapper;
}
export declare namespace Method_a_decimals_out {
    type Implementation = Func_a_decimals_out.Implementation;
    type Kwargs = {
        decimals?: number;
        out?: NDArray<any> | null;
    };
    type Wrapper = (decimals: number | Kwargs, out?: NDArray<any> | null | Kwargs) => NDArray<any>;
    const decorator: ({ defaults, implementation, parsers, this_as_first_arg }: {
        defaults: [string, any][];
        implementation: Func_a_decimals_out.Implementation;
        parsers?: ((kwargs: any) => void)[];
        this_as_first_arg?: boolean;
    }) => Wrapper;
    const defaults: [string, any][];
    const parsers: ((kwargs: any) => void)[];
    const defaultDecorator: (implementation: Implementation) => Wrapper;
}
type NDArray_ = NDArray<any> | number | boolean;
export declare namespace Func_a_other_out {
    type Implementation = (a: NDArray<any>, other: NDArray<any>, out: NDArray<any> | null) => NDArray<any>;
    type Kwargs = {
        a: NDArray_;
        other?: NDArray_;
        out?: NDArray<any> | null;
    };
    type Wrapper = (a: NDArray_ | Kwargs, other: NDArray_ | Kwargs, out?: NDArray<any> | null | Kwargs) => NDArray<any>;
    const decorator: ({ defaults, implementation, parsers, this_as_first_arg }: {
        defaults: [string, any][];
        implementation: Implementation;
        parsers?: ((kwargs: any) => void)[];
        this_as_first_arg?: boolean;
    }) => Wrapper;
    const defaults: [string, any][];
    const parsers: ((kwargs: any) => void)[];
    const defaultDecorator: (implementation: Implementation) => Wrapper;
}
export declare namespace Method_other_out {
    type Implementation = Func_a_other_out.Implementation;
    type Kwargs = {
        other?: NDArray_;
        out?: NDArray<any> | null;
    };
    type Wrapper<T extends TypedArrayConstructor = Float64ArrayConstructor> = (other: NDArray_ | Kwargs, out?: NDArray<any> | null | Kwargs) => NDArray<T>;
    const defaults: [string, any][];
    const parsers: ((kwargs: any) => void)[];
    function defaultDecorator<T extends TypedArrayConstructor = Float64ArrayConstructor>(implementation: Implementation): Wrapper<T>;
}
export declare namespace Func_a_values_where {
    type Implementation = (a: NDArray<any>, values: NDArray<any>, where: Where) => NDArray<any>;
    type Kwargs = {
        a: NDArray<any>;
        values?: NDArray_;
        where: Where;
    };
    type Wrapper = (a: NDArray<any> | Kwargs, values: NDArray_ | Kwargs, ...where: Where) => NDArray<any>;
    const decorator: ({ defaults, implementation, parsers, this_as_first_arg }: {
        defaults: [string, any][];
        implementation: Implementation;
        parsers?: ((kwargs: any) => void)[];
        this_as_first_arg?: boolean;
    }) => Wrapper;
    const defaults: [string, any][];
    const parsers: ((kwargs: any) => void)[];
    const defaultDecorator: (implementation: Implementation) => Wrapper;
}
export declare namespace Method_values_where {
    type Implementation = Func_a_values_where.Implementation;
    type Kwargs = {
        values?: NDArray_;
        where: Where;
    };
    type Wrapper = (values: NDArray_ | Kwargs, ...where: Where) => NDArray<any>;
    const decorator: ({ defaults, implementation, parsers, this_as_first_arg }: {
        defaults: [string, any][];
        implementation: Func_a_values_where.Implementation;
        parsers?: ((kwargs: any) => void)[];
        this_as_first_arg?: boolean;
    }) => Wrapper;
    const defaults: [string, any][];
    const parsers: ((kwargs: any) => void)[];
    const defaultDecorator: (implementation: Implementation) => Wrapper;
}
export declare namespace Func_a_axis_keepdims {
    type Implementation = (a: NDArray<any>, axis: number, keepdims: boolean) => NDArray<any>;
    type Kwargs = {
        a?: NDArray_;
        axis?: AxisArg;
        keepdims?: boolean;
    };
    type Wrapper<T extends TypedArrayConstructor = Float64ArrayConstructor> = (a: NDArray_ | Kwargs, axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs) => NDArray<T>;
    const decorator: ({ defaults, implementation, parsers, this_as_first_arg }: {
        defaults: [string, any][];
        implementation: Implementation;
        parsers?: ((kwargs: any) => void)[];
        this_as_first_arg?: boolean;
    }) => Wrapper<Float64ArrayConstructor>;
    const defaults: [string, any][];
    const parsers: ((kwargs: any) => void)[];
    function defaultDecorator<T extends TypedArrayConstructor = Float64ArrayConstructor>(implementation: Implementation): Wrapper<T>;
}
export declare namespace Method_a_axis_keepdims {
    type Implementation = Func_a_axis_keepdims.Implementation;
    type Kwargs = {
        axis?: AxisArg;
        keepdims?: boolean;
    };
    type Wrapper<T extends TypedArrayConstructor = Float64ArrayConstructor> = (axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs) => NDArray<T>;
    const decorator: ({ defaults, implementation, parsers, this_as_first_arg }: {
        defaults: [string, any][];
        implementation: Func_a_axis_keepdims.Implementation;
        parsers?: ((kwargs: any) => void)[];
        this_as_first_arg?: boolean;
    }) => Wrapper<Float64ArrayConstructor>;
    const defaults: [string, any][];
    const parsers: ((kwargs: any) => void)[];
    function defaultDecorator<T extends TypedArrayConstructor = Float64ArrayConstructor>(implementation: Implementation): Wrapper<T>;
}
export declare namespace Func_a_ord_axis_keepdims {
    type Implementation = (a: NDArray<any>, ord: number, axis: number, keepdims: boolean) => NDArray<any>;
    type Kwargs = {
        a?: NDArray_;
        ord?: number;
        axis?: AxisArg;
        keepdims?: boolean;
    };
    type Wrapper = (a: NDArray_ | Kwargs, ord?: number | Kwargs, axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs) => NDArray<any>;
    const decorator: ({ defaults, implementation, parsers, this_as_first_arg }: {
        defaults: [string, any][];
        implementation: Implementation;
        parsers?: ((kwargs: any) => void)[];
        this_as_first_arg?: boolean;
    }) => Wrapper;
    const defaults: [string, any][];
    const parsers: ((kwargs: any) => void)[];
    function defaultDecorator(implementation: Implementation): Wrapper;
}
export declare namespace Method_a_ord_axis_keepdims {
    type Kwargs = {
        ord?: number;
        axis?: AxisArg;
        keepdims?: boolean;
    };
    type Wrapper = (ord?: number | Kwargs, axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs) => NDArray<any>;
    type Implementation = Func_a_ord_axis_keepdims.Implementation;
    function defaultDecorator(implementation: Implementation): Wrapper;
}
export declare namespace Func_a_axis_ddof_keepdims {
    type Implementation = (a: NDArray<any>, axis: number, ddof: number, keepdims: boolean) => NDArray<any>;
    type Kwargs = {
        a?: NDArray_;
        axis?: AxisArg;
        ddof?: number;
        keepdims?: boolean;
    };
    type Wrapper = (a: NDArray_ | Kwargs, axis?: AxisArg | Kwargs, ddof?: number | Kwargs, keepdims?: boolean | Kwargs) => NDArray<any>;
    const decorator: ({ defaults, implementation, parsers, this_as_first_arg }: {
        defaults: [string, any][];
        implementation: Implementation;
        parsers?: ((kwargs: any) => void)[];
        this_as_first_arg?: boolean;
    }) => Wrapper;
    const defaults: [string, any][];
    const parsers: ((kwargs: any) => void)[];
    function defaultDecorator(implementation: Implementation): Wrapper;
}
export declare namespace Method_a_axis_ddof_keepdims {
    type Implementation = Func_a_axis_ddof_keepdims.Implementation;
    type Kwargs = {
        ddof?: number;
        axis?: AxisArg;
        keepdims?: boolean;
    };
    type Wrapper = (ddof?: number | Kwargs, axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs) => NDArray<any>;
    function defaultDecorator(implementation: Implementation): Wrapper;
}
export declare namespace Func_y_x_out {
    type Implementation = (y: NDArray_, x: NDArray_, out?: NDArray | null) => NDArray<any>;
    type Kwargs = {
        y?: NDArray_;
        x?: NDArray_;
        out?: NDArray<any> | null;
    };
    type Wrapper = (y: NDArray_ | Kwargs, x: NDArray_ | Kwargs, out?: NDArray<any> | null | Kwargs) => NDArray<any>;
    const decorator: ({ defaults, implementation, parsers, this_as_first_arg }: {
        defaults: [string, any][];
        implementation: Implementation;
        parsers?: ((kwargs: any) => void)[];
        this_as_first_arg?: boolean;
    }) => Wrapper;
    const defaults: [string, any][];
    const parsers: ((kwargs: any) => void)[];
    const defaultDecorator: (implementation: Implementation) => Wrapper;
}
export {};
//# sourceMappingURL=kwargs.d.ts.map
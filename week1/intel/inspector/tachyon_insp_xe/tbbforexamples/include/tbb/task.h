/*

!==============================================================
!
! SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
! http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
!
! Copyright 2016 Intel Corporation
!
! THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
! NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
! PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
!
! =============================================================
*/

#ifndef __TBB_task_H
#define __TBB_task_H

#include "tbb_stddef.h"
#include "tbb_machine.h"
#include <climits>

typedef struct ___itt_caller *__itt_caller;

namespace tbb {

class task;
class task_list;

#if __TBB_TASK_GROUP_CONTEXT
class task_group_context;
#endif /* __TBB_TASK_GROUP_CONTEXT */

// MSVC does not allow taking the address of a member that was defined
// privately in task_base and made public in class task via a using declaration.
#if _MSC_VER || (__GNUC__==3 && __GNUC_MINOR__<3)
#define __TBB_TASK_BASE_ACCESS public
#else
#define __TBB_TASK_BASE_ACCESS private
#endif

namespace internal {

    class allocate_additional_child_of_proxy: no_assign {
        //! No longer used, but retained for binary layout compatibility.  Always NULL.
        task* self;
        task& parent;
    public:
        explicit allocate_additional_child_of_proxy( task& parent_ ) : self(NULL), parent(parent_) {}
        task& __TBB_EXPORTED_METHOD allocate( size_t size ) const;
        void __TBB_EXPORTED_METHOD free( task& ) const;
    };

}

namespace interface5 {
    namespace internal {
        //! Base class for methods that became static in TBB 3.0.
        /** TBB's evolution caused the "this" argument for several methods to become obsolete.
            However, for backwards binary compatibility, the new methods need distinct names,
            otherwise the One Definition Rule would be broken.  Hence the new methods are
            defined in this private base class, and then exposed in class task via
            using declarations. */
        class task_base: tbb::internal::no_copy {
        __TBB_TASK_BASE_ACCESS:
            friend class tbb::task;

            //! Schedule task for execution when a worker becomes available.
            static void spawn( task& t );

            //! Spawn multiple tasks and clear list.
            static void spawn( task_list& list );

            //! Like allocate_child, except that task's parent becomes "t", not this.
            /** Typically used in conjunction with schedule_to_reexecute to implement while loops.
               Atomically increments the reference count of t.parent() */
            static tbb::internal::allocate_additional_child_of_proxy allocate_additional_child_of( task& t ) {
                return tbb::internal::allocate_additional_child_of_proxy(t);
            }

            //! Destroy a task.
            /** Usually, calling this method is unnecessary, because a task is
                implicitly deleted after its execute() method runs.  However,
                sometimes a task needs to be explicitly deallocated, such as
                when a root task is used as the parent in spawn_and_wait_for_all. */
            static void __TBB_EXPORTED_FUNC destroy( task& victim );
        };
    } // internal
} // interface5

//! @cond INTERNAL
namespace internal {

    class scheduler: no_copy {
    public:
        //! For internal use only
        virtual void spawn( task& first, task*& next ) = 0;

        //! For internal use only
        virtual void wait_for_all( task& parent, task* child ) = 0;

        //! For internal use only
        virtual void spawn_root_and_wait( task& first, task*& next ) = 0;

        //! Pure virtual destructor;
        //  Have to have it just to shut up overzealous compilation warnings
        virtual ~scheduler() = 0;

        //! For internal use only
        virtual void enqueue( task& t, void* reserved ) = 0;
    };

    //! A reference count
    /** Should always be non-negative.  A signed type is used so that underflow can be detected. */
    typedef intptr_t reference_count;

    //! An id as used for specifying affinity.
    typedef unsigned short affinity_id;

#if __TBB_TASK_GROUP_CONTEXT
    class generic_scheduler;

    struct context_list_node_t {
        context_list_node_t *my_prev,
                            *my_next;
    };

    class allocate_root_with_context_proxy: no_assign {
        task_group_context& my_context;
    public:
        allocate_root_with_context_proxy ( task_group_context& ctx ) : my_context(ctx) {}
        task& __TBB_EXPORTED_METHOD allocate( size_t size ) const;
        void __TBB_EXPORTED_METHOD free( task& ) const;
    };
#endif /* __TBB_TASK_GROUP_CONTEXT */

    class allocate_root_proxy: no_assign {
    public:
        static task& __TBB_EXPORTED_FUNC allocate( size_t size );
        static void __TBB_EXPORTED_FUNC free( task& );
    };

    class allocate_continuation_proxy: no_assign {
    public:
        task& __TBB_EXPORTED_METHOD allocate( size_t size ) const;
        void __TBB_EXPORTED_METHOD free( task& ) const;
    };

    class allocate_child_proxy: no_assign {
    public:
        task& __TBB_EXPORTED_METHOD allocate( size_t size ) const;
        void __TBB_EXPORTED_METHOD free( task& ) const;
    };

    //! Memory prefix to a task object.
    /** This class is internal to the library.
        Do not reference it directly, except within the library itself.
        Fields are ordered in way that preserves backwards compatibility and yields
        good packing on typical 32-bit and 64-bit platforms.

        In case task prefix size exceeds 32 or 64 bytes on IA32 and Intel64
        architectures correspondingly, consider dynamic setting of task_alignment
        and task_prefix_reservation_size based on the maximal operand size supported
        by the current CPU.

        @ingroup task_scheduling */
    class task_prefix {
    private:
        friend class tbb::task;
        friend class tbb::interface5::internal::task_base;
        friend class tbb::task_list;
        friend class internal::scheduler;
        friend class internal::allocate_root_proxy;
        friend class internal::allocate_child_proxy;
        friend class internal::allocate_continuation_proxy;
        friend class internal::allocate_additional_child_of_proxy;

#if __TBB_TASK_GROUP_CONTEXT
        //! Shared context that is used to communicate asynchronous state changes
        /** Currently it is used to broadcast cancellation requests generated both
            by users and as the result of unhandled exceptions in the task::execute()
            methods. */
        task_group_context  *context;
#endif /* __TBB_TASK_GROUP_CONTEXT */

        //! The scheduler that allocated the task, or NULL if the task is big.
        /** Small tasks are pooled by the scheduler that allocated the task.
            If a scheduler needs to free a small task allocated by another scheduler,
            it returns the task to that other scheduler.  This policy avoids
            memory space blowup issues for memory allocators that allocate from
            thread-specific pools. */
        scheduler* origin;

#if __TBB_TASK_PRIORITY
        union {
#endif /* __TBB_TASK_PRIORITY */
        //! Obsolete. The scheduler that owns the task.
        /** Retained only for the sake of backward binary compatibility.
            Still used by inline methods in the task.h header. **/
        scheduler* owner;

#if __TBB_TASK_PRIORITY
        //! Pointer to the next offloaded lower priority task.
        /** Used to maintain a list of offloaded tasks inside the scheduler. **/
        task* next_offloaded;
        };
#endif /* __TBB_TASK_PRIORITY */

        //! The task whose reference count includes me.
        /** In the "blocking style" of programming, this field points to the parent task.
            In the "continuation-passing style" of programming, this field points to the
            continuation of the parent. */
        tbb::task* parent;

        //! Reference count used for synchronization.
        /** In the "continuation-passing style" of programming, this field is
            the difference of the number of allocated children minus the
            number of children that have completed.
            In the "blocking style" of programming, this field is one more than the difference. */
        __TBB_atomic reference_count ref_count;

        //! Obsolete. Used to be scheduling depth before TBB 2.2
        /** Retained only for the sake of backward binary compatibility.
            Not used by TBB anymore. **/
        int depth;

        //! A task::state_type, stored as a byte for compactness.
        /** This state is exposed to users via method task::state(). */
        unsigned char state;

        //! Miscellaneous state that is not directly visible to users, stored as a byte for compactness.
        /** 0x0 -> version 1.0 task
            0x1 -> version >=2.1 task
            0x10 -> task was enqueued 
            0x20 -> task_proxy
            0x40 -> task has live ref_count
            0x80 -> a stolen task */
        unsigned char extra_state;

        affinity_id affinity;

        //! "next" field for list of task
        tbb::task* next;

        //! The task corresponding to this task_prefix.
        tbb::task& task() {return *reinterpret_cast<tbb::task*>(this+1);}
    };

} // namespace internal
//! @endcond

#if __TBB_TASK_GROUP_CONTEXT

#if __TBB_TASK_PRIORITY
namespace internal {
    static const int priority_stride_v4 = INT_MAX / 4;
}

enum priority_t {
    priority_normal = internal::priority_stride_v4 * 2,
    priority_low = priority_normal - internal::priority_stride_v4,
    priority_high = priority_normal + internal::priority_stride_v4
};

#endif /* __TBB_TASK_PRIORITY */

#if TBB_USE_CAPTURED_EXCEPTION
    class tbb_exception;
#else
    namespace internal {
        class tbb_exception_ptr;
    }
#endif /* !TBB_USE_CAPTURED_EXCEPTION */

class task_scheduler_init;

//! Used to form groups of tasks
/** @ingroup task_scheduling
    The context services explicit cancellation requests from user code, and unhandled
    exceptions intercepted during tasks execution. Intercepting an exception results
    in generating internal cancellation requests (which is processed in exactly the
    same way as external ones).

    The context is associated with one or more root tasks and defines the cancellation
    group that includes all the descendants of the corresponding root task(s). Association
    is established when a context object is passed as an argument to the task::allocate_root()
    method. See task_group_context::task_group_context for more details.

    The context can be bound to another one, and other contexts can be bound to it,
    forming a tree-like structure: parent -> this -> children. Arrows here designate
    cancellation propagation direction. If a task in a cancellation group is cancelled
    all the other tasks in this group and groups bound to it (as children) get cancelled too.

    IMPLEMENTATION NOTE:
    When adding new members to task_group_context or changing types of existing ones,
    update the size of both padding buffers (_leading_padding and _trailing_padding)
    appropriately. See also VERSIONING NOTE at the constructor definition below. **/
class task_group_context : internal::no_copy {
private:
    friend class internal::generic_scheduler;
    friend class task_scheduler_init;

#if TBB_USE_CAPTURED_EXCEPTION
    typedef tbb_exception exception_container_type;
#else
    typedef internal::tbb_exception_ptr exception_container_type;
#endif

    enum version_traits_word_layout {
        traits_offset = 16,
        version_mask = 0xFFFF,
        traits_mask = 0xFFFFul << traits_offset
    };

public:
    enum kind_type {
        isolated,
        bound
    };

    enum traits_type {
        exact_exception = 0x0001ul << traits_offset,
        concurrent_wait = 0x0004ul << traits_offset,
#if TBB_USE_CAPTURED_EXCEPTION
        default_traits = 0
#else
        default_traits = exact_exception
#endif /* !TBB_USE_CAPTURED_EXCEPTION */
    };

private:
    enum state {
        may_have_children = 1
    };

    union {
        //! Flavor of this context: bound or isolated.
        kind_type my_kind;
        uintptr_t _my_kind_aligner;
    };

    //! Pointer to the context of the parent cancellation group. NULL for isolated contexts.
    task_group_context *my_parent;

    //! Used to form the thread specific list of contexts without additional memory allocation.
    /** A context is included into the list of the current thread when its binding to
        its parent happens. Any context can be present in the list of one thread only. **/
    internal::context_list_node_t my_node;

    //! Used to set and maintain stack stitching point for Intel Performance Tools.
    __itt_caller itt_caller;

    //! Leading padding protecting accesses to frequently used members from false sharing.
    /** Read accesses to the field my_cancellation_requested are on the hot path inside
        the scheduler. This padding ensures that this field never shares the same cache
        line with a local variable that is frequently written to. **/
    char _leading_padding[internal::NFS_MaxLineSize
                          - 2 * sizeof(uintptr_t)- sizeof(void*) - sizeof(internal::context_list_node_t)
                          - sizeof(__itt_caller)];

    //! Specifies whether cancellation was request for this task group.
    uintptr_t my_cancellation_requested;

    //! Version for run-time checks and behavioral traits of the context.
    /** Version occupies low 16 bits, and traits (zero or more ORed enumerators
        from the traits_type enumerations) take the next 16 bits.
        Original (zeroth) version of the context did not support any traits. **/
    uintptr_t  my_version_and_traits;

    //! Pointer to the container storing exception being propagated across this task group.
    exception_container_type *my_exception;

    //! Scheduler instance that registered this context in its thread specific list.
    internal::generic_scheduler *my_owner;

    //! Internal state (combination of state flags).
    uintptr_t my_state;

#if __TBB_TASK_PRIORITY
    //! Priority level of the task group (in normalized representation)
    intptr_t my_priority;
#endif /* __TBB_TASK_PRIORITY */

    //! Trailing padding protecting accesses to frequently used members from false sharing
    /** \sa _leading_padding **/
    char _trailing_padding[internal::NFS_MaxLineSize - 2 * sizeof(uintptr_t) - 2 * sizeof(void*)
#if __TBB_TASK_PRIORITY
                            - sizeof(intptr_t)
#endif /* __TBB_TASK_PRIORITY */
                          ];

public:
    //! Default & binding constructor.
    /** By default a bound context is created. That is this context will be bound
        (as child) to the context of the task calling task::allocate_root(this_context)
        method. Cancellation requests passed to the parent context are propagated
        to all the contexts bound to it. Similarly priority change is propagated
        from the parent context to its children.

        If task_group_context::isolated is used as the argument, then the tasks associated
        with this context will never be affected by events in any other context.

        Creating isolated contexts involve much less overhead, but they have limited
        utility. Normally when an exception occurs in an algorithm that has nested
        ones running, it is desirably to have all the nested algorithms cancelled
        as well. Such a behavior requires nested algorithms to use bound contexts.

        There is one good place where using isolated algorithms is beneficial. It is
        a master thread. That is if a particular algorithm is invoked directly from
        the master thread (not from a TBB task), supplying it with explicitly
        created isolated context will result in a faster algorithm startup.

        VERSIONING NOTE:
        Implementation(s) of task_group_context constructor(s) cannot be made
        entirely out-of-line because the run-time version must be set by the user
        code. This will become critically important for binary compatibility, if
        we ever have to change the size of the context object.

        Boosting the runtime version will also be necessary if new data fields are
        introduced in the currently unused padding areas and these fields are updated
        by inline methods. **/
    task_group_context ( kind_type relation_with_parent = bound,
                         uintptr_t traits = default_traits )
        : my_kind(relation_with_parent)
        , my_version_and_traits(1 | traits)
    {
        init();
    }

    __TBB_EXPORTED_METHOD ~task_group_context ();

    //! Forcefully reinitializes the context after the task tree it was associated with is completed.
    /** Because the method assumes that all the tasks that used to be associated with
        this context have already finished, calling it while the context is still
        in use somewhere in the task hierarchy leads to undefined behavior.

        IMPORTANT: This method is not thread safe!

        The method does not change the context's parent if it is set. **/
    void __TBB_EXPORTED_METHOD reset ();

    //! Initiates cancellation of all tasks in this cancellation group and its subordinate groups.
    /** \return false if cancellation has already been requested, true otherwise.

        Note that canceling never fails. When false is returned, it just means that
        another thread (or this one) has already sent cancellation request to this
        context or to one of its ancestors (if this context is bound). It is guaranteed
        that when this method is concurrently called on the same not yet cancelled
        context, true will be returned by one and only one invocation. **/
    bool __TBB_EXPORTED_METHOD cancel_group_execution ();

    //! Returns true if the context received cancellation request.
    bool __TBB_EXPORTED_METHOD is_group_execution_cancelled () const;

    //! Records the pending exception, and cancels the task group.
    /** May be called only from inside a catch-block. If the context is already
        cancelled, does nothing.
        The method brings the task group associated with this context exactly into
        the state it would be in, if one of its tasks threw the currently pending
        exception during its execution. In other words, it emulates the actions
        of the scheduler's dispatch loop exception handler. **/
    void __TBB_EXPORTED_METHOD register_pending_exception ();

#if __TBB_TASK_PRIORITY
    //! Changes priority of the task group
    void set_priority ( priority_t );

    //! Retrieves current priority of the current task group
    priority_t priority () const;
#endif /* __TBB_TASK_PRIORITY */

protected:
    //! Out-of-line part of the constructor.
    /** Singled out to ensure backward binary compatibility of the future versions. **/
    void __TBB_EXPORTED_METHOD init ();

private:
    friend class task;
    friend class internal::allocate_root_with_context_proxy;

    static const kind_type binding_required = bound;
    static const kind_type binding_completed = kind_type(bound+1);
    static const kind_type detached = kind_type(binding_completed+1);
    static const kind_type dying = kind_type(detached+1);

    //! Propagates state change (if any) from an ancestor
    /** Checks if one of this object's ancestors is in a new state, and propagates
        the new state to all its descendants in this object's heritage line. **/
    template <typename T>
    void propagate_state_from_ancestors ( T task_group_context::*mptr_state, T new_state );

    //! Makes sure that the context is registered with a scheduler instance.
    inline void finish_initialization ( internal::generic_scheduler *local_sched );

    //! Registers this context with the local scheduler and binds it to its parent context
    void bind_to ( internal::generic_scheduler *local_sched );

    //! Registers this context with the local scheduler
    void register_with ( internal::generic_scheduler *local_sched );

}; // class task_group_context

#endif /* __TBB_TASK_GROUP_CONTEXT */

//! Base class for user-defined tasks.
/** @ingroup task_scheduling */
class task: __TBB_TASK_BASE_ACCESS interface5::internal::task_base {

    //! Set reference count
    void __TBB_EXPORTED_METHOD internal_set_ref_count( int count );

    //! Decrement reference count and return its new value.
    internal::reference_count __TBB_EXPORTED_METHOD internal_decrement_ref_count();

protected:
    //! Default constructor.
    task() {prefix().extra_state=1;}

public:
    //! Destructor.
    virtual ~task() {}

    //! Should be overridden by derived classes.
    virtual task* execute() = 0;

    //! Enumeration of task states that the scheduler considers.
    enum state_type {
        //! task is running, and will be destroyed after method execute() completes.
        executing,
        //! task to be rescheduled.
        reexecute,
        //! task is in ready pool, or is going to be put there, or was just taken off.
        ready,
        //! task object is freshly allocated or recycled.
        allocated,
        //! task object is on free list, or is going to be put there, or was just taken off.
        freed,
        //! task to be recycled as continuation
        recycle
    };

    //------------------------------------------------------------------------
    // Allocating tasks
    //------------------------------------------------------------------------

    //! Returns proxy for overloaded new that allocates a root task.
    static internal::allocate_root_proxy allocate_root() {
        return internal::allocate_root_proxy();
    }

#if __TBB_TASK_GROUP_CONTEXT
    //! Returns proxy for overloaded new that allocates a root task associated with user supplied context.
    static internal::allocate_root_with_context_proxy allocate_root( task_group_context& ctx ) {
        return internal::allocate_root_with_context_proxy(ctx);
    }
#endif /* __TBB_TASK_GROUP_CONTEXT */

    //! Returns proxy for overloaded new that allocates a continuation task of *this.
    /** The continuation's parent becomes the parent of *this. */
    internal::allocate_continuation_proxy& allocate_continuation() {
        return *reinterpret_cast<internal::allocate_continuation_proxy*>(this);
    }

    //! Returns proxy for overloaded new that allocates a child task of *this.
    internal::allocate_child_proxy& allocate_child() {
        return *reinterpret_cast<internal::allocate_child_proxy*>(this);
    }

    //! Define recommended static form via import from base class.
    using task_base::allocate_additional_child_of;

#if __TBB_DEPRECATED_TASK_INTERFACE
    //! Destroy a task.
    /** Usually, calling this method is unnecessary, because a task is
        implicitly deleted after its execute() method runs.  However,
        sometimes a task needs to be explicitly deallocated, such as
        when a root task is used as the parent in spawn_and_wait_for_all. */
    void __TBB_EXPORTED_METHOD destroy( task& t );
#else /* !__TBB_DEPRECATED_TASK_INTERFACE */
    //! Define recommended static form via import from base class.
    using task_base::destroy;
#endif /* !__TBB_DEPRECATED_TASK_INTERFACE */

    //------------------------------------------------------------------------
    // Recycling of tasks
    //------------------------------------------------------------------------

    //! Change this to be a continuation of its former self.
    /** The caller must guarantee that the task's refcount does not become zero until
        after the method execute() returns.  Typically, this is done by having
        method execute() return a pointer to a child of the task.  If the guarantee
        cannot be made, use method recycle_as_safe_continuation instead.

        Because of the hazard, this method may be deprecated in the future. */
    void recycle_as_continuation() {
        __TBB_ASSERT( prefix().state==executing, "execute not running?" );
        prefix().state = allocated;
    }

    //! Recommended to use, safe variant of recycle_as_continuation
    /** For safety, it requires additional increment of ref_count.
        With no descendants and ref_count of 1, it has the semantics of recycle_to_reexecute. */
    void recycle_as_safe_continuation() {
        __TBB_ASSERT( prefix().state==executing, "execute not running?" );
        prefix().state = recycle;
    }

    //! Change this to be a child of new_parent.
    void recycle_as_child_of( task& new_parent ) {
        internal::task_prefix& p = prefix();
        __TBB_ASSERT( prefix().state==executing||prefix().state==allocated, "execute not running, or already recycled" );
        __TBB_ASSERT( prefix().ref_count==0, "no child tasks allowed when recycled as a child" );
        __TBB_ASSERT( p.parent==NULL, "parent must be null" );
        __TBB_ASSERT( new_parent.prefix().state<=recycle, "corrupt parent's state" );
        __TBB_ASSERT( new_parent.prefix().state!=freed, "parent already freed" );
        p.state = allocated;
        p.parent = &new_parent;
#if __TBB_TASK_GROUP_CONTEXT
        p.context = new_parent.prefix().context;
#endif /* __TBB_TASK_GROUP_CONTEXT */
    }

    //! Schedule this for reexecution after current execute() returns.
    /** Made obsolete by recycle_as_safe_continuation; may become deprecated. */
    void recycle_to_reexecute() {
        __TBB_ASSERT( prefix().state==executing, "execute not running, or already recycled" );
        __TBB_ASSERT( prefix().ref_count==0, "no child tasks allowed when recycled for reexecution" );
        prefix().state = reexecute;
    }

    // All depth-related methods are obsolete, and are retained for the sake
    // of backward source compatibility only
    intptr_t depth() const {return 0;}
    void set_depth( intptr_t ) {}
    void add_to_depth( int ) {}


    //------------------------------------------------------------------------
    // Spawning and blocking
    //------------------------------------------------------------------------

    //! Set reference count
    void set_ref_count( int count ) {
#if TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT
        internal_set_ref_count(count);
#else
        prefix().ref_count = count;
#endif /* TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT */
    }

    //! Atomically increment reference count and returns its old value.
    /** Has acquire semantics */
    void increment_ref_count() {
        __TBB_FetchAndIncrementWacquire( &prefix().ref_count );
    }

    //! Atomically decrement reference count and returns its new value.
    /** Has release semantics. */
    int decrement_ref_count() {
#if TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT
        return int(internal_decrement_ref_count());
#else
        return int(__TBB_FetchAndDecrementWrelease( &prefix().ref_count ))-1;
#endif /* TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT */
    }

    //! Define recommended static forms via import from base class.
    using task_base::spawn;

    //! Similar to spawn followed by wait_for_all, but more efficient.
    void spawn_and_wait_for_all( task& child ) {
        prefix().owner->wait_for_all( *this, &child );
    }

    //! Similar to spawn followed by wait_for_all, but more efficient.
    void __TBB_EXPORTED_METHOD spawn_and_wait_for_all( task_list& list );

    //! Spawn task allocated by allocate_root, wait for it to complete, and deallocate it.
    static void spawn_root_and_wait( task& root ) {
        root.prefix().owner->spawn_root_and_wait( root, root.prefix().next );
    }

    //! Spawn root tasks on list and wait for all of them to finish.
    /** If there are more tasks than worker threads, the tasks are spawned in
        order of front to back. */
    static void spawn_root_and_wait( task_list& root_list );

    //! Wait for reference count to become one, and set reference count to zero.
    /** Works on tasks while waiting. */
    void wait_for_all() {
        prefix().owner->wait_for_all( *this, NULL );
    }

    //! Enqueue task for starvation-resistant execution.
#if __TBB_TASK_PRIORITY
    /** The task will be enqueued on the normal priority level disregarding the
        priority of its task group.

        The rationale of such semantics is that priority of an enqueued task is
        statically fixed at the moment of its enqueuing, while task group priority
        is dynamic. Thus automatic priority inheritance would be generally a subject
        to the race, which may result in unexpected behavior.

        Use enqueue() overload with explicit priority value and task::group_priority()
        method to implement such priority inheritance when it is really necessary. **/
#endif /* __TBB_TASK_PRIORITY */
    static void enqueue( task& t ) {
        t.prefix().owner->enqueue( t, NULL );
    }

#if __TBB_TASK_PRIORITY
    //! Enqueue task for starvation-resistant execution on the specified priority level.
    static void enqueue( task& t, priority_t p ) {
        __TBB_ASSERT( p == priority_low || p == priority_normal || p == priority_high, "Invalid priority level value" );
        t.prefix().owner->enqueue( t, (void*)p );
    }
#endif /* __TBB_TASK_PRIORITY */

    //! The innermost task being executed or destroyed by the current thread at the moment.
    static task& __TBB_EXPORTED_FUNC self();

    //! task on whose behalf this task is working, or NULL if this is a root.
    task* parent() const {return prefix().parent;}

    //! sets parent task pointer to specified value
    void set_parent(task* p) {
#if __TBB_TASK_GROUP_CONTEXT
        __TBB_ASSERT(prefix().context == p->prefix().context, "The tasks must be in the same context");
#endif
        prefix().parent = p;
    }

#if __TBB_TASK_GROUP_CONTEXT
    //! This method is deprecated and will be removed in the future.
    /** Use method group() instead. **/
    task_group_context* context() {return prefix().context;}

    //! Pointer to the task group descriptor.
    task_group_context* group () { return prefix().context; }
#endif /* __TBB_TASK_GROUP_CONTEXT */

    //! True if task was stolen from the task pool of another thread.
    bool is_stolen_task() const {
        return (prefix().extra_state & 0x80)!=0;
    }

    //------------------------------------------------------------------------
    // Debugging
    //------------------------------------------------------------------------

    //! Current execution state
    state_type state() const {return state_type(prefix().state);}

    //! The internal reference count.
    int ref_count() const {
#if TBB_USE_ASSERT
        internal::reference_count ref_count_ = prefix().ref_count;
        __TBB_ASSERT( ref_count_==int(ref_count_), "integer overflow error");
#endif
        return int(prefix().ref_count);
    }

    //! Obsolete, and only retained for the sake of backward compatibility. Always returns true.
    bool __TBB_EXPORTED_METHOD is_owned_by_current_thread() const;

    //------------------------------------------------------------------------
    // Affinity
    //------------------------------------------------------------------------

    //! An id as used for specifying affinity.
    /** Guaranteed to be integral type.  Value of 0 means no affinity. */
    typedef internal::affinity_id affinity_id;

    //! Set affinity for this task.
    void set_affinity( affinity_id id ) {prefix().affinity = id;}

    //! Current affinity of this task
    affinity_id affinity() const {return prefix().affinity;}

    //! Invoked by scheduler to notify task that it ran on unexpected thread.
    /** Invoked before method execute() runs, if task is stolen, or task has
        affinity but will be executed on another thread.

        The default action does nothing. */
    virtual void __TBB_EXPORTED_METHOD note_affinity( affinity_id id );

#if __TBB_TASK_GROUP_CONTEXT
    //! Moves this task from its current group into another one.
    /** Argument ctx specifies the new group.

        The primary purpose of this method is to associate unique task group context
        with a task allocated for subsequent enqueuing. In contrast to spawned tasks
        enqueued ones normally outlive the scope where they were created. This makes
        traditional usage model where task group context are allocated locally on
        the stack inapplicable. Dynamic allocation of context objects is performance
        inefficient. Method change_group() allows to make task group context object
        a member of the task class, and then associate it with its containing task
        object in the latter's constructor. **/
    void __TBB_EXPORTED_METHOD change_group ( task_group_context& ctx );

    //! Initiates cancellation of all tasks in this cancellation group and its subordinate groups.
    /** \return false if cancellation has already been requested, true otherwise. **/
    bool cancel_group_execution () { return prefix().context->cancel_group_execution(); }

    //! Returns true if the context has received cancellation request.
    bool is_cancelled () const { return prefix().context->is_group_execution_cancelled(); }
#endif /* __TBB_TASK_GROUP_CONTEXT */

#if __TBB_TASK_PRIORITY
    //! Changes priority of the task group this task belongs to.
    void set_group_priority ( priority_t p ) {  prefix().context->set_priority(p); }

    //! Retrieves current priority of the task group this task belongs to.
    priority_t group_priority () const { return prefix().context->priority(); }

#endif /* __TBB_TASK_PRIORITY */

private:
    friend class interface5::internal::task_base;
    friend class task_list;
    friend class internal::scheduler;
    friend class internal::allocate_root_proxy;
#if __TBB_TASK_GROUP_CONTEXT
    friend class internal::allocate_root_with_context_proxy;
#endif /* __TBB_TASK_GROUP_CONTEXT */
    friend class internal::allocate_continuation_proxy;
    friend class internal::allocate_child_proxy;
    friend class internal::allocate_additional_child_of_proxy;

    //! Get reference to corresponding task_prefix.
    /** Version tag prevents loader on Linux from using the wrong symbol in debug builds. **/
    internal::task_prefix& prefix( internal::version_tag* = NULL ) const {
        return reinterpret_cast<internal::task_prefix*>(const_cast<task*>(this))[-1];
    }
}; // class task

//! task that does nothing.  Useful for synchronization.
/** @ingroup task_scheduling */
class empty_task: public task {
    /*override*/ task* execute() {
        return NULL;
    }
};

//! A list of children.
/** Used for method task::spawn_children
    @ingroup task_scheduling */
class task_list: internal::no_copy {
private:
    task* first;
    task** next_ptr;
    friend class task;
    friend class interface5::internal::task_base;
public:
    //! Construct empty list
    task_list() : first(NULL), next_ptr(&first) {}

    //! Destroys the list, but does not destroy the task objects.
    ~task_list() {}

    //! True if list if empty; false otherwise.
    bool empty() const {return !first;}

    //! Push task onto back of list.
    void push_back( task& task ) {
        task.prefix().next = NULL;
        *next_ptr = &task;
        next_ptr = &task.prefix().next;
    }

    //! Pop the front task from the list.
    task& pop_front() {
        __TBB_ASSERT( !empty(), "attempt to pop item from empty task_list" );
        task* result = first;
        first = result->prefix().next;
        if( !first ) next_ptr = &first;
        return *result;
    }

    //! Clear the list
    void clear() {
        first=NULL;
        next_ptr=&first;
    }
};

inline void interface5::internal::task_base::spawn( task& t ) {
    t.prefix().owner->spawn( t, t.prefix().next );
}

inline void interface5::internal::task_base::spawn( task_list& list ) {
    if( task* t = list.first ) {
        t->prefix().owner->spawn( *t, *list.next_ptr );
        list.clear();
    }
}

inline void task::spawn_root_and_wait( task_list& root_list ) {
    if( task* t = root_list.first ) {
        t->prefix().owner->spawn_root_and_wait( *t, *root_list.next_ptr );
        root_list.clear();
    }
}

} // namespace tbb

inline void *operator new( size_t bytes, const tbb::internal::allocate_root_proxy& ) {
    return &tbb::internal::allocate_root_proxy::allocate(bytes);
}

inline void operator delete( void* task, const tbb::internal::allocate_root_proxy& ) {
    tbb::internal::allocate_root_proxy::free( *static_cast<tbb::task*>(task) );
}

#if __TBB_TASK_GROUP_CONTEXT
inline void *operator new( size_t bytes, const tbb::internal::allocate_root_with_context_proxy& p ) {
    return &p.allocate(bytes);
}

inline void operator delete( void* task, const tbb::internal::allocate_root_with_context_proxy& p ) {
    p.free( *static_cast<tbb::task*>(task) );
}
#endif /* __TBB_TASK_GROUP_CONTEXT */

inline void *operator new( size_t bytes, const tbb::internal::allocate_continuation_proxy& p ) {
    return &p.allocate(bytes);
}

inline void operator delete( void* task, const tbb::internal::allocate_continuation_proxy& p ) {
    p.free( *static_cast<tbb::task*>(task) );
}

inline void *operator new( size_t bytes, const tbb::internal::allocate_child_proxy& p ) {
    return &p.allocate(bytes);
}

inline void operator delete( void* task, const tbb::internal::allocate_child_proxy& p ) {
    p.free( *static_cast<tbb::task*>(task) );
}

inline void *operator new( size_t bytes, const tbb::internal::allocate_additional_child_of_proxy& p ) {
    return &p.allocate(bytes);
}

inline void operator delete( void* task, const tbb::internal::allocate_additional_child_of_proxy& p ) {
    p.free( *static_cast<tbb::task*>(task) );
}

#endif /* __TBB_task_H */

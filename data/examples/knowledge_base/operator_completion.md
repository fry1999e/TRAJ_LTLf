# Operator Completion

## Boundary-driven X/F/G completion

In Traj-LTLf, restricted skeletons are completed into full LTLf formulas by
inserting X, F, and G only after finite traces have been reviewed.

Boundary traces should have the highest priority during completion because they
distinguish subtle temporal meanings:

- F phi is true at position i if phi holds at i or at some later position.
- X phi is strong next. It is false at the last position of a finite trace.
- X F phi requires a real next position and then eventually phi from that next
  position onward.
- G phi checks phi at every position from the current point to the end of the
  finite trace.

For response requirements such as "whenever A occurs, B should eventually
occur", a useful hole template is:

H_scope(A -> H_response(B))

where:

- H_scope in {epsilon, G}
- H_response in {epsilon, X, F, X F}

Boundary traces decide whether simultaneous A and B is allowed. If a one-step
boundary trace {A, B} is labeled positive, F B should be preferred over X F B,
because F allows the current position while strong X requires a next position.

Positive and negative traces remain important as auxiliary constraints. They
ensure that the completed formula still accepts all positive traces and rejects
all negative traces, so short but under-specified formulas are not selected only
because they match a boundary case.

## Nested temporal holes

If a generated template prematurely places X, F, or G inside an existing hole,
the concrete operator is converted into a unique typed child hole instead of
being deleted or causing the whole batch to fail.

- A temporal wrapper in a trigger or other non-response subformula becomes
  `H_scope_child_N`.
- A temporal wrapper in the response branch that is not an exact match for the
  main response becomes `H_response_child_N`.
- Nested chains remain nested, for example `X(F(a))` becomes
  `H_scope_child_1(H_scope_child_2(a))`.

Each child hole records its structural role, parent hole, and the recovered
operator as a non-binding source hint. Only the root `H_scope` receives global
scope heuristics automatically.

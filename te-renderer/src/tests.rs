use crate::instances::rangetree::RangeTree;

#[test]
fn remove_left_half_rangetree() {
    let mut rt = RangeTree::new();
    for i in 0..100 {
        rt.add_num(i)
    }

    for i in 0..50 {
        rt.remove_num(i)
    }

    let result = rt.get_vec();
    dbg!(&result);
    assert_eq!(result, vec![50..100]);
}

#[test]
fn remove_right_half_rangetree() {
    let mut rt = RangeTree::new();
    for i in 0..100 {
        rt.add_num(i)
    }

    for i in 50..99 {
        rt.remove_num(i)
    }

    let result = rt.get_vec();
    dbg!(&result);
    rt.remove_num(99);
    let result = rt.get_vec();
    dbg!(&result);
    assert_eq!(result, vec![0..50]);
}

#[test]
fn remove_mid_rangetree() {
    let mut rt = RangeTree::new();
    for i in 0..100 {
        rt.add_num(i)
    }

    rt.remove_num(50);
    let result = rt.get_vec();
    dbg!(&result);
    assert_eq!(result, vec![0..50, 51..100]);
}

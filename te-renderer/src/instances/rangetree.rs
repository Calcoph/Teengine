use std::ops::Range;

struct RangeTree { // TODO: Make this more efficient
    first_node: Option<RangeNode>
}

impl RangeTree {
    fn new() -> Self {
        Self { first_node: None }
    }

    fn add_num(&mut self, num: usize) {
        match &mut self.first_node {
            Some(node) => node.add_num(num),
            None => self.first_node = Some(RangeNode::new(num)),
        }
    }

    fn clean(&mut self) {
        self.first_node = None
    }
}

struct RangeNode {
    contents: Range<usize>,
    left: Option<Box<RangeNode>>,
    right: Option<Box<RangeNode>>
}

impl RangeNode {
    fn new(num: usize) -> Self {
        RangeNode {
            contents: num..num+1,
            left: None,
            right: None
        }
    }

    fn add_num(&mut self, num: usize) {
        if self.contents.end == num {
            match &self.right {
                Some(node) => {
                    if node.contents.start == num-1 {
                        let mut right = self.right.take().unwrap();
                        let new_right = right.right.take();
                        self.right = new_right;
                        self.contents.end = right.contents.end;
                    } else {
                        self.contents.end = num+1
                    }
                },
                None => self.contents.end = num+1,
            }
        } else if self.contents.start-1 == num {
            match &self.left {
                Some(node) => {
                    if node.contents.end == num {
                        let mut left = self.left.take().unwrap();
                        let new_left = left.left.take();
                        self.left = new_left;
                        self.contents.start = left.contents.start;
                    } else {
                        self.contents.start = num
                    }
                },
                None => self.contents.start = num,
            }
        } else if num > self.contents.end {
            match &mut self.right {
                Some(node) => node.add_num(num),
                None => self.right = Some(Box::new(RangeNode::new(num))),
            }
        } else { // num < self.contents.start-1
            match &mut self.left {
                Some(node) => node.add_num(num),
                None => self.left = Some(Box::new(RangeNode::new(num))),
            }
        }
    }
}

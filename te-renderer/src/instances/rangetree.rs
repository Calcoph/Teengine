use std::ops::Range;

#[derive(Debug)]
pub(crate) struct RangeTree {
    // TODO: Make this more efficient
    // TODO: Make it self-balanced
    first_node: Option<RangeNode>
}

impl RangeTree {
    pub(crate) fn new() -> Self {
        Self { first_node: None }
    }

    pub(crate) fn add_num(&mut self, num: u32) {
        match &mut self.first_node {
            Some(node) => node.add_num(num),
            None => self.first_node = Some(RangeNode::new(num)),
        }
    }

    pub(crate) fn clean(&mut self) {
        self.first_node = None
    }

    pub(crate) fn contains(&self, num: &u32) -> bool {
        match &self.first_node {
            Some(node) => node.contains(num),
            None => false,
        }
    }

    pub(crate) fn get_vec(&self) -> Vec<Range<u32>> {
        match &self.first_node {
            Some(node) => node.get_vec(),
            None => Vec::new(),
        }
    }

    pub(crate) fn remove_num(&mut self, num: u32) {
        match &mut self.first_node {
            Some(node) => {
                let (is_removed, new_node) = node.remove_num(num);
                if is_removed {
                    self.first_node = match new_node {
                        Some(node) => Some(*node),
                        None => None,
                    }
                }
            },
            None => (),
        }
    }
}

/* #[derive(Debug)] // TODO: Use this
enum AVLValue {
    Left,
    Right,
    Balanced
} */

#[derive(Debug)]
struct RangeNode {
    contents: Range<u32>,
    left: Option<Box<RangeNode>>,
    right: Option<Box<RangeNode>>,
    //avl_value: AVLValue
}

impl RangeNode {
    fn new(num: u32) -> Self {
        RangeNode {
            contents: num..num+1,
            left: None,
            right: None,
            //avl_value: AVLValue::Balanced
        }
    }

    fn add_num(&mut self, num: u32) {
        if self.contents.end == num {
            match &self.right {
                Some(node) => {
                    if num > 0 && node.contents.start == num-1 {
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
        } else if self.contents.start > 0 && self.contents.start-1 == num {
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
        } else if num < self.contents.start-1 {
            match &mut self.left {
                Some(node) => node.add_num(num),
                None => self.left = Some(Box::new(RangeNode::new(num))),
            }
        } else {
            unreachable!()
        }
    }

    fn contains(&self, num: &u32) -> bool {
        self.contents.contains(num)
        || match &self.left {
            Some(node) => node.contains(num),
            None => false,
        }
        || match &self.right {
            Some(node) => node.contains(num),
            None => false,
        }
    }

    fn get_vec(&self) -> Vec<Range<u32>> {
        let mut v = match &self.left {
            Some(node) => {
                let mut v = node.get_vec();
                v.push(self.contents.clone());
                v
            },
            None => vec![self.contents.clone()],
        };

        if let Some(node) = &self.right {
            node.expand_vec(&mut v)
        };

        v
    }

    fn expand_vec(&self, v: &mut Vec<Range<u32>>) {
        if let Some(node) = &self.left {
            node.expand_vec(v)
        };

        v.push(self.contents.clone());

        if let Some(node) = &self.right {
            node.expand_vec(v)
        };
    }

    /// true if this node should be deleted. In that case replace it with Option<Self>
    fn remove_num(&mut self, num: u32) -> (bool, Option<Box<Self>>) {
        if self.contents.start == num {
            if self.contents.end == num+1 {
                let left = self.left.take();
                match self.right.take() {
                    Some(mut right) => {
                        right.place_left(left);
                        (true, Some(right))
                    },
                    None => (true, left),
                }
            } else {
                self.contents.start = num+1;
                (false, None)
            }
        } else if self.contents.end == num+1 {
            self.contents.end = num;
            (false, None)
        } else if self.contents.start > num {
            match &mut self.left {
                Some(node) => {
                    let (is_removed, new_node) = node.remove_num(num);
                    if is_removed {
                        self.left = new_node
                    }
                },
                None => (),
            };
            (false, None)
        } else if self.contents.end-1 < num {
            match &mut self.right {
                Some(node) => {
                    let (is_removed, new_node) = node.remove_num(num);
                    if is_removed {
                        self.right = new_node
                    }
                },
                None => (),
            };
            (false, None)
        } else { // num is in the middle of the range
            let left = self.contents.start..num;
            let right = num+1..self.contents.end;
            match &self.left {
                Some(node) => match &mut self.right {
                    Some(node) => {
                        self.contents = left;
                        for i in right {
                            node.add_num(i) // TODO: Optimize this, so the range isn't converted to int to be converted to range again
                        }
                    },
                    None => {
                        self.right = Some(Box::new(RangeNode {
                            contents: right,
                            left: None,
                            right: None,
                        }));
                        self.contents = left;
                    },
                },
                None => {
                    self.left = Some(Box::new(RangeNode {
                        contents: left,
                        left: None,
                        right: None,
                    }));
                    self.contents = right;
                },
            };
            (false, None)
        }
    }

    fn place_left(&mut self, left: Option<Box<RangeNode>>) {
        match &mut self.left {
            Some(node) => node.place_left(left),
            None => self.left = left,
        }
    }
}
